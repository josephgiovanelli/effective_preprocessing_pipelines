from experiment.pipeline.PrototypeSingleton import PrototypeSingleton
from experiment.policies.policy import Policy
from experiment.algorithm import space as ALGORITHM_SPACE
from experiment.objective import objective_pipeline, objective_algo

import functools

from hyperopt import tpe, fmin, Trials


class Split(Policy):

    def run(self, X, y):
        super(Split, self).run(X, y)
        current_pipeline_configuration = {}
        current_algo_configuration = {}
        trials_pipelines = Trials()
        trials_algo = Trials()

        if self.config["experiment"] == "preprocessing_impact":
            if self.config['time'] - self.config['step_pipeline'] > 0:
                current_algo_configuration = self._optimize_algorithm(X, y, current_pipeline_configuration, trials_algo)
            if self.config['step_pipeline'] > 0:
                self._optimize_pipeline(X, y, current_algo_configuration, trials_pipelines)
        else:
            if self.config['step_pipeline'] > 0:
                current_pipeline_configuration = self._optimize_pipeline(X, y, current_algo_configuration, trials_pipelines)
            if self.config['time'] - self.config['step_pipeline'] > 0:
                self._optimize_algorithm(X, y, current_pipeline_configuration, trials_algo)

    def _optimize_algorithm(self, X, y, current_pipeline_configuration, trials_algo):
        print('## Algorithm')
        obj_algo = functools.partial(objective_algo,
                current_pipeline_config=current_pipeline_configuration,
                algorithm=self.config['algorithm'],
                X=X,
                y=y,
                context=self.context,
                config=self.config)
        fmin(fn=obj_algo,
            space=ALGORITHM_SPACE.get_domain_space(self.config['algorithm']),
            algo=tpe.suggest,
            max_evals=75 if self.config["experiment"] == "preprocessing_impact" else None,
            max_time=80000 if self.config["experiment"] == "preprocessing_impact" else (self.config['time'] - self.config['step_pipeline']),
            trials=trials_algo,
            show_progressbar=False,
            verbose=0
        )

        best_config = self.context['best_config']
        current_algo_configuration = best_config['algorithm']
        super(Split, self).display_step_results(best_config)
        return current_algo_configuration

    def _optimize_pipeline(self, X, y, current_algo_configuration, trials_pipelines):
        print('## Data Pipeline')
        obj_pl = functools.partial(objective_pipeline,
                current_algo_config=current_algo_configuration,
                algorithm=self.config['algorithm'],
                X=X,
                y=y,
                context=self.context,
                config=self.config)
        fmin(
            fn=obj_pl, 
            space=self.PIPELINE_SPACE,
            algo=tpe.suggest, 
            max_evals=75 if self.config["experiment"] == "preprocessing_impact" else None,
            max_time=80000 if self.config["experiment"] == "preprocessing_impact" else self.config['step_pipeline'],
            trials=trials_pipelines,
            show_progressbar=False,
            verbose=0
        )

        best_config = self.context['best_config']
        current_pipeline_configuration = best_config['pipeline']
        super(Split, self).display_step_results(best_config)
        return current_pipeline_configuration
