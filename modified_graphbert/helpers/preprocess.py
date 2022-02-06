from abc import ABC
import os
import logging
from graph_bert.MethodWLNodeColoring import MethodWLNodeColoring
from graph_bert.MethodGraphBatching import MethodGraphBatching
from graph_bert.MethodHopDistance import MethodHopDistance
from graph_bert.ResultSaving import ResultSaving
from graph_bert.Settings import Settings

os.environ["CUDA_VISIBLE_DEVICES"] = "5"


class AbstractModel(ABC):
    def __init__(self, data_obj, dataset_name):
        self.dataset_name = dataset_name
        self.data_obj = data_obj

    def run(self, method_obj, result_destination_folder_path, evaluate_obj=None, k=None):
        result_obj = ResultSaving()
        result_obj.result_destination_folder_path = result_destination_folder_path
        if k:
            if "Batch" in result_destination_folder_path:
                result_obj.result_destination_file_name = self.dataset_name + '_' + str(k)
            elif "Hop" in result_destination_folder_path:
                result_obj.result_destination_file_name = 'hop_' + self.dataset_name + '_' + str(k)
            else:
                result_obj.result_destination_file_name = self.dataset_name
        else:
            result_obj.result_destination_file_name = self.dataset_name

        setting_obj = Settings()
        setting_obj.prepare(self.data_obj, method_obj, result_obj, evaluate_obj)
        setting_obj.load_run_save_evaluate()


class Preprocessor(AbstractModel):
    def __init__(self, data_obj, dataset_name):
        super().__init__(data_obj, dataset_name)
        self.dataset_name = dataset_name
        self.data_obj = data_obj

    def run_wl(self):
        wl_coloring = MethodWLNodeColoring()
        self.run(wl_coloring, './result/WL/')
        logging.info(f'************ Finished WL coloring ************')

    def run_graph_batching(self, k_list=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50)):
        for k in k_list:
            graph_batching = MethodGraphBatching()
            graph_batching.k = k
            self.run(graph_batching, './result/Batch/', k=k)
            logging.info(f'************ Finished Graph Batching at {k} ************')

    def run_hop_distance(self, max_k=10):
        for k in range(1, max_k + 1):
            method_obj = MethodHopDistance()
            method_obj.k = k
            method_obj.dataset_name = self.dataset_name
            self.run(method_obj, './result/Hop/', k=k)
            logging.info(f'************ Finished Hop Distance at {k} ************')
