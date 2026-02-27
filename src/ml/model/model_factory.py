import sys
import importlib
from typing import List, Tuple
from collections import namedtuple
from sklearn.metrics import accuracy_score, f1_score
import yaml

from src.exception import CustomerException
from src.logger import logging


InitializedModelDetail = namedtuple("InitializedModelDetail",
                                     ["model_serial_number", "model", "param_grid_search", "model_name"])

GridSearchedBestModel = namedtuple("GridSearchedBestModel", ["model_serial_number",
                                                               "model",
                                                               "best_model",
                                                               "best_parameters",
                                                               "best_score"])

BestModel = namedtuple("BestModel", ["model_serial_number",
                                      "model",
                                      "best_model",
                                      "best_parameters",
                                      "best_score"])


class ModelFactory:
    def __init__(self, model_config_path: str):
        try:
            self.config: dict = ModelFactory.read_params(model_config_path)
            self.grid_search_cv_module: str = self.config['grid_search']['module']
            self.grid_search_class_name: str = self.config['grid_search']['class']
            self.grid_search_property_data: dict = dict(self.config['grid_search']['params'])
            self.models_initialization_config: dict = dict(self.config['model_selection'])
            self.initialized_model_list = None
            self.grid_searched_best_model_list = None

        except Exception as e:
            raise CustomerException(e, sys) from e

    @staticmethod
    def read_params(config_path: str) -> dict:
        try:
            with open(config_path) as yaml_file:
                config: dict = yaml.safe_load(yaml_file)
            return config
        except Exception as e:
            raise CustomerException(e, sys) from e

    @staticmethod
    def class_for_name(module_name: str, class_name: str):
        try:
            # load the module, will raise ImportError if module cannot be loaded
            module = importlib.import_module(module_name)
            # get the class, will raise AttributeError if class cannot be found
            logging.info(f"Executing command: from {module} import {class_name}")
            class_ref = getattr(module, class_name)
            return class_ref
        except Exception as e:
            raise CustomerException(e, sys) from e

    @staticmethod
    def update_property_of_class(instance_ref: object, property_data: dict):
        try:
            if not isinstance(property_data, dict):
                raise Exception("property_data parameter required to dictionary")
            print(property_data)
            for key, value in property_data.items():
                logging.info(f"Executing: $ {str(instance_ref)}.{key}={value}")
                setattr(instance_ref, key, value)
            return instance_ref
        except Exception as e:
            raise CustomerException(e, sys) from e

    def get_initialized_model_list(self) -> List[InitializedModelDetail]:
        try:
            initialized_model_list = []
            for model_serial_number in self.models_initialization_config.keys():
                model_initialization_config = self.models_initialization_config[model_serial_number]
                model_obj_ref = ModelFactory.class_for_name(module_name=model_initialization_config['module'],
                                                             class_name=model_initialization_config['class'])
                model = model_obj_ref()

                if 'params' in model_initialization_config:
                    model_obj_property_data = dict(model_initialization_config['params'])
                    model = ModelFactory.update_property_of_class(instance_ref=model,
                                                                   property_data=model_obj_property_data)

                param_grid_search = model_initialization_config['search_param_grid']
                model_name = f"{model_initialization_config['module']}.{model_initialization_config['class']}"

                model_initialization_config = InitializedModelDetail(model_serial_number=model_serial_number,
                                                                       model=model,
                                                                       param_grid_search=param_grid_search,
                                                                       model_name=model_name)

                initialized_model_list.append(model_initialization_config)

            self.initialized_model_list = initialized_model_list
            return initialized_model_list
        except Exception as e:
            raise CustomerException(e, sys) from e

    def execute_grid_search_operation(self, initialized_model: InitializedModelDetail, input_feature,
                                        output_feature) -> GridSearchedBestModel:
        try:
            # Instantiated GridSearchCV class
            grid_search_cv_ref = ModelFactory.class_for_name(module_name=self.grid_search_cv_module,
                                                               class_name=self.grid_search_class_name)

            grid_search_cv = grid_search_cv_ref(estimator=initialized_model.model,
                                                  param_grid=initialized_model.param_grid_search)
            grid_search_cv = ModelFactory.update_property_of_class(grid_search_cv,
                                                                     self.grid_search_property_data)

            message = f'{">>" * 30} Training {type(initialized_model.model).__name__} {"<<" * 30}'
            logging.info(message)
            grid_search_cv.fit(input_feature, output_feature)
            grid_searched_best_model = GridSearchedBestModel(model_serial_number=initialized_model.model_serial_number,
                                                              model=initialized_model.model,
                                                              best_model=grid_search_cv.best_estimator_,
                                                              best_parameters=grid_search_cv.best_params_,
                                                              best_score=grid_search_cv.best_score_)
            return grid_searched_best_model
        except Exception as e:
            raise CustomerException(e, sys) from e

    def initiate_best_parameter_search_for_initialized_model(self, initialized_model: InitializedModelDetail,
                                                               input_feature,
                                                               output_feature) -> GridSearchedBestModel:
        try:
            return self.execute_grid_search_operation(initialized_model=initialized_model,
                                                       input_feature=input_feature,
                                                       output_feature=output_feature)
        except Exception as e:
            raise CustomerException(e, sys) from e

    def initiate_best_parameter_search_for_initialized_models(self,
                                                                initialized_model_list: List[InitializedModelDetail],
                                                                input_feature,
                                                                output_feature) -> List[GridSearchedBestModel]:
        try:
            self.grid_searched_best_model_list = []
            for initialized_model in initialized_model_list:
                grid_searched_best_model = self.initiate_best_parameter_search_for_initialized_model(
                    initialized_model=initialized_model,
                    input_feature=input_feature,
                    output_feature=output_feature)
                self.grid_searched_best_model_list.append(grid_searched_best_model)
            return self.grid_searched_best_model_list
        except Exception as e:
            raise CustomerException(e, sys) from e

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list: List[GridSearchedBestModel],
                                                            base_accuracy=0.6) -> BestModel:
        try:
            best_model = None
            for grid_searched_best_model in grid_searched_best_model_list:
                if base_accuracy < grid_searched_best_model.best_score:
                    logging.info(f"Acceptable model found: {grid_searched_best_model}")
                    base_accuracy = grid_searched_best_model.best_score

                    best_model = grid_searched_best_model
            if not best_model:
                raise Exception(f"None of the models achieved accuracy greater than the base accuracy: {base_accuracy}")
            logging.info(f"Best model: {best_model}")
            return best_model
        except Exception as e:
            raise CustomerException(e, sys) from e

    def get_best_model(self, X, y, base_accuracy=0.6) -> BestModel:
        try:
            logging.info("Started Initializing models from config file")
            initialized_model_list = self.get_initialized_model_list()
            logging.info(f"Initialized model list: {initialized_model_list}")

            logging.info("Started grid search operation")
            grid_searched_best_model_list = self.initiate_best_parameter_search_for_initialized_models(
                initialized_model_list=initialized_model_list,
                input_feature=X,
                output_feature=y
            )

            return ModelFactory.get_best_model_from_grid_searched_best_model_list(
                grid_searched_best_model_list=grid_searched_best_model_list,
                base_accuracy=base_accuracy
            )
        except Exception as e:
            raise CustomerException(e, sys) from e
