import logging
import os
import traceback
from pickle import PicklingError

import billiard
from billiard.exceptions import SoftTimeLimitExceeded, WorkerLostError
from billiard.pool import MaybeEncodingError
from celery import Celery
from celery.exceptions import CeleryError
from celery.signals import after_setup_logger

import projectconfig
from relnet.environment.graph_edge_env import GraphEdgeEnv
from relnet.utils.config_utils import get_logger_instance, date_format, logging_format
from relnet.utils.general_utils import create_generator_instance

app_settings = projectconfig.get_project_config()

app = Celery('tasks',
             backend='amqp://',
             broker=app_settings.CELERY_BROKER_URL)

app.conf.update(accept_content=['application/x-python-serialize', 'application/json'],
                task_serializer=app_settings.CELERY_TASK_SERIALIZER,
                result_serializer=app_settings.CELERY_RESULT_SERIALIZER,
                task_acks_late=app_settings.CELERY_TASK_ACKS_LATE,
                worker_prefetch_multiplier=app_settings.CELERYD_PREFETCH_MULTIPLIER,
                worker_max_tasks_per_child=app_settings.WORKER_MAX_TASKS_PER_CHILD,
                worker_concurrency=app_settings.get_number_worker_threads(),
                broker_heartbeat=app_settings.BROKER_HEARTBEAT,
                broker_pool_limit=app_settings.BROKER_POOL_LIMIT,
                timezone='Europe/London',
                enable_utc=False)


@after_setup_logger.connect
def setup_loggers(logger, *args, **kwargs):
    formatter = logging.Formatter(fmt=logging_format, datefmt=date_format)
    fh = logging.FileHandler(f"/tmp/celery-{os.getenv('HOSTNAME')}.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


class DistributedTaskException(Exception):
    pass


ExpectedErrors = (FileNotFoundError,
                  ValueError,
                  RuntimeError,
                  SystemError,
                  ConnectionResetError,

                  WorkerLostError,
                  CeleryError,
                  billiard.pool.MaybeEncodingError,
                  PicklingError)

@app.task(bind=True,
          rate_limit="60/m",
          retry_kwargs={'max_retries': 15},
          autoretry_for=(DistributedTaskException,),
          retry_backoff=True,
          retry_jitter=True)
def optimize_hyperparams_task(self,
                              agent,
                              objective_function,
                              network_generator,
                              experiment_conditions,
                              file_paths,
                              hyperparams,
                              hyperparams_id,
                              is_only_hyp_comb,
                              model_seed,
                              model_identifier_prefix,
                              train_kwargs=None,
                              eval_make_action_kwargs=None,
                              additional_opts=None):


    gen_params = experiment_conditions.gen_params

    network_generator_instance = create_generator_instance(network_generator, file_paths)

    models_dir = file_paths.models_dir
    timings_dir = file_paths.timings_dir

    obj_fun_kwargs = {"random_seed": experiment_conditions.obj_fun_seed,
                      "num_mc_sims_multiplier": experiment_conditions.num_mc_sims_multiplier}

    env = GraphEdgeEnv(objective_function(), obj_fun_kwargs, experiment_conditions.possible_edge_percentage,
                       conn_radius_modifier=network_generator.conn_radius_modifiers[experiment_conditions.restriction_mechanism],
                       restriction_mechanism=experiment_conditions.restriction_mechanism)
    agent_instance = agent(env)

    run_options = {}

    run_options["random_seed"] = model_seed
    run_options["models_path"] = models_dir
    run_options["timings_path"] = timings_dir

    run_options["log_progress"] = True

    log_filename = str(file_paths.construct_log_filepath())
    run_options["log_filename"] = log_filename
    run_options["model_identifier_prefix"] = model_identifier_prefix

    run_options["restore_model"] = False
    run_options["use_geometric_features"] = experiment_conditions.use_geometric_features

    run_options["log_timings"] = False

    run_options.update((additional_opts or {}))
    agent_instance.setup(run_options, hyperparams)

    if is_only_hyp_comb:
        average_reward = 1.0
        if file_paths.hyperopt_results_dir is not None:
            hyperopt_result_file = f"{file_paths.hyperopt_results_dir.absolute()}/" + \
                                   file_paths.construct_best_validation_file_name(model_identifier_prefix)
            hyperopt_result_out = open(hyperopt_result_file, 'w')
            hyperopt_result_out.write('%.6f\n' % (average_reward))
            hyperopt_result_out.close()

    else:
        try:
            validation_graphs = network_generator_instance.generate_many(gen_params, experiment_conditions.validation_seeds)

            if agent.is_trainable:
                train_graphs = network_generator_instance.generate_many(gen_params, experiment_conditions.train_seeds)
                max_steps = experiment_conditions.agent_budgets[objective_function.name][agent.algorithm_name]

                agent_train_kwargs =  (train_kwargs or {})
                try:
                    agent_instance.train(train_graphs, validation_graphs, max_steps, **agent_train_kwargs)
                except TypeError:
                    print(f"the graph at fault is from gen {network_generator.name} with seed {experiment_conditions.train_seeds[0]}!")

            agent_eval_kwargs = (eval_make_action_kwargs or {})
            average_reward = agent_instance.eval(validation_graphs, make_action_kwargs=agent_eval_kwargs)

            if file_paths.hyperopt_results_dir is not None:
                hyperopt_result_file = f"{file_paths.hyperopt_results_dir.absolute()}/" + \
                                       file_paths.construct_best_validation_file_name(model_identifier_prefix)
                hyperopt_result_out = open(hyperopt_result_file, 'w')
                hyperopt_result_out.write('%.6f\n' % (average_reward))
                hyperopt_result_out.close()

            agent_instance.finalize()

        except ExpectedErrors as error:
            raise DistributedTaskException() from error

    return hyperparams, objective_function.name, network_generator_instance.name, average_reward

@app.task(bind=True,
          rate_limit="1000/m",
          soft_time_limit=24 * 60 * 60,
          time_limit=25 * 60 * 60,
          autoretry_for=(DistributedTaskException,),
          retry_kwargs={'max_retries': 15},
          retry_backoff=True,
          retry_jitter=True)
def evaluate_for_network_seed_task(self,
                                   agent,
                                   objective_function,
                                   network_generator,
                                   best_hyperparams,
                                   best_hyperparams_id,
                                   experiment_conditions,
                                   file_paths,
                                   net_seed,
                                   model_seeds,
                                   graph_id=None,
                                   eval_make_action_kwargs=None,
                                   additional_opts=None):

    try:
        log_filename = str(file_paths.construct_log_filepath())
        logger = get_logger_instance(log_filename)


        local_results = []

        gen_params = experiment_conditions.gen_params
        network_generator_instance = create_generator_instance(network_generator, file_paths)

        models_dir = file_paths.models_dir
        timings_dir = file_paths.timings_dir

        obj_fun_kwargs = {"random_seed": experiment_conditions.obj_fun_seed,
                          "num_mc_sims_multiplier": experiment_conditions.num_mc_sims_multiplier}

        env = GraphEdgeEnv(objective_function(), obj_fun_kwargs, experiment_conditions.possible_edge_percentage,
                           conn_radius_modifier=network_generator.conn_radius_modifiers[
                               experiment_conditions.restriction_mechanism],
                           restriction_mechanism=experiment_conditions.restriction_mechanism)

        for model_seed in model_seeds:
            try:
                agent_instance = agent(env)

                run_options = {}
                run_options['random_seed'] = model_seed
                run_options["restore_model"] = True

                model_identifier_prefix = file_paths.construct_model_identifier_prefix(agent.algorithm_name,
                                                                                       objective_function.name,
                                                                                       network_generator_instance.name,
                                                                                       model_seed,
                                                                                       best_hyperparams_id,
                                                                                       graph_id=graph_id)
                run_options["model_identifier_prefix"] = model_identifier_prefix
                run_options["models_path"] = models_dir
                run_options["timings_path"] = timings_dir

                run_options["log_progress"] = True
                run_options["log_filename"] = log_filename
                run_options["use_geometric_features"] = experiment_conditions.use_geometric_features

                run_options["log_timings"] = True

                run_options.update((additional_opts or {}))
                agent_instance.setup(run_options, best_hyperparams)

                result_row = {}
                result_row['network_generator'] = network_generator_instance.name
                if graph_id is not None:
                    result_row['graph_id'] = graph_id

                result_row['objective_function'] = objective_function.name
                result_row['network_seed'] = net_seed

                result_row['algorithm'] = agent.algorithm_name
                result_row['agent_seed'] = model_seed
                result_row['network_size'] = gen_params['n']

                test_g_list = [network_generator_instance.generate(gen_params, net_seed)]
                agent_eval_kwargs = (eval_make_action_kwargs or {})
                result_row['cummulative_reward'] = agent_instance.eval(test_g_list, make_action_kwargs=agent_eval_kwargs)

                local_results.append(result_row)

                agent_instance.finalize()

            except ExpectedErrors as error:
                logger.warn("faced the following exception:")
                logger.warn(traceback.format_exc(limit=100))
                raise DistributedTaskException() from error

        return local_results
    except SoftTimeLimitExceeded:
        logger.warn(f"Task with id {self.request.id} went over the time limit. aborting...")
        return []


@app.task(bind=True,
          soft_time_limit=24 * 60 * 60,
          time_limit=25 * 60 * 60,
          autoretry_for=(DistributedTaskException,),
          retry_kwargs={'max_retries': 5},
          retry_backoff=True,
          retry_jitter=True)
def compute_property_for_network_seed(self,
                                   objective_function,
                                   network_generator,
                                   experiment_conditions,
                                   file_paths,
                                   net_seed):

    try:
        log_filename = str(file_paths.construct_log_filepath())
        logger = get_logger_instance(log_filename)
        gen_params = experiment_conditions.gen_params
        obj_fun_kwargs = {"random_seed": experiment_conditions.obj_fun_seed,
                          "num_mc_sims": experiment_conditions.num_mc_sims}

        gen_instance = network_generator()

        try:
            graph_instance = gen_instance.generate(gen_params, net_seed)
            result_row = {}
            result_row['network_generator'] = network_generator.name
            result_row['objective_function'] = objective_function.name
            result_row['network_seed'] = net_seed
            result_row['network_size'] = gen_params['n']
            result_row['value'] = objective_function.compute(graph_instance, **obj_fun_kwargs)




        except ExpectedErrors as error:
            logger.warn("faced the following exception:")
            logger.warn(traceback.format_exc(limit=100))
            raise DistributedTaskException() from error

        return result_row
    except SoftTimeLimitExceeded:
        logger.warn(f"Task with id {self.request.id} went over the time limit. aborting...")
        return []






























































