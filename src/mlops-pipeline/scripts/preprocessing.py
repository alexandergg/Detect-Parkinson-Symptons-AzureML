import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import distutils.util
import seaborn as sns
import matplotlib.pyplot as plt

from azureml.core.run import Run, _OfflineRun
from sklearn.decomposition import PCA
from pandas_profiling import ProfileReport
from sklearn.manifold import TSNE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.model_selection import train_test_split
from azureml.core import Workspace, Datastore, Dataset, Experiment
from sklearn.feature_selection import mutual_info_classif, SelectKBest, SelectPercentile


class ExploratoryAnalysis():

    def __init__(self):
        self.__parser = argparse.ArgumentParser("preprocessing")
        self.__parser.add_argument("--datastore", type=str,
                                   help="Name of the datastore",
                                   default="workspaceblobstore")
        self.__parser.add_argument("--dataset_name", type=str,
                                   help="Name of the dataset")
        self.__parser.add_argument("--dataset_preprocessed_name", type=str,
                                   help="Standard preprocessed dataset")
        self.__parser.add_argument("--output_preprocess_dataset", type=str,
                                   help="Name of the PipelineData reference")

        self.__args = self.__parser.parse_args()
        self.__run = Run.get_context()
        self.__local_run = type(self.__run) == _OfflineRun

        if self.__local_run:
            self.__ws = Workspace.from_config('../../notebooks-settings')
            self.__exp = Experiment(self.__ws, 'exploratory_analysis')
            self.__run = self.__exp.start_logging()
        else:
            self.__ws = self.__run.experiment.workspace
            self.__exp = self.__run.experiment

        self.__datastore = Datastore.get(
            self.__ws, datastore_name=self.__args.datastore)

    def main(self):
        df, df_eda = self.__preprocess_dataset(schema_path="./schema_dataset.json")
        self.__make_exploratory_analysis(df_eda)
        self.__upload_datasets(df, df.columns)

    def __preprocess_dataset(self, schema_path):
        with open(schema_path) as f:
            schema = json.load(f)

        df, df_eda = self.__get_dataset(self.__args.dataset_name)
        columns_names = schema.keys()
        df.columns = columns_names

        return df, df_eda

    def __make_exploratory_analysis(self, df):
        self.__frequency_tremor(df)
        self.__tremor_acceleration_energy(df)
        self.__execute_tsne(df)

    def __get_dataset(self, dataset_name):
        acc0 = self.__ws.datasets.get(f"{dataset_name}0_dataset").to_pandas_dataframe()
        acc1 = self.__ws.datasets.get(f"{dataset_name}1_dataset").to_pandas_dataframe()
        acc2 = self.__ws.datasets.get(f"{dataset_name}2_dataset").to_pandas_dataframe()
        acc3 = self.__ws.datasets.get(f"{dataset_name}3_dataset").to_pandas_dataframe()

        df_eda = pd.concat([acc0,acc1,acc2,acc3],axis=0)
        df_eda['Tremor'] = df_eda['Tremor'].replace(2,1)

        df = pd.concat([acc0,acc1,acc2,acc3],axis=0)
        df['Tremor'] = df['Tremor'].replace(2,1)

        return df, df_eda

    def __upload_datasets(self, df, columns):
            dataset_name, preprocess_filepath, datastore_path = self.__get_dataset_metadata(
                df, "train")
            self.__upload_dataset(self.__ws, self.__datastore, dataset_name,
                                  datastore_path, preprocess_filepath,
                                  use_datadrift=False, type_dataset="standard")

    def __get_dataset_metadata(self, df, extension):
        dataset_name = f'{self.__args.dataset_preprocessed_name}_{extension}'
        output_preprocessed_directory = self.__args.output_preprocess_dataset if extension == "train" else f'{self.__args.output_preprocess_dataset}_{extension}'
        preprocess_filepath = os.path.join(output_preprocessed_directory,
                                           f'{dataset_name}.csv')
        datastore_path = f"parkinson/{dataset_name}.csv"

        os.makedirs(output_preprocessed_directory, exist_ok=True)
        df.to_csv(preprocess_filepath, index=False)

        return dataset_name, preprocess_filepath, datastore_path

    def __upload_dataset(self, ws, def_blob_store, dataset_name, datastore_path, filepath, use_datadrift, type_dataset):
        def_blob_store.upload_files(
            [filepath], target_path="parkinson", overwrite=True)
        tab_data_set = Dataset.Tabular.from_delimited_files(
            path=(def_blob_store, datastore_path))
        try:
            tab_data_set.register(workspace=ws,
                                  name=f'{dataset_name}',
                                  description=f'{dataset_name} data',
                                  tags={'format': 'CSV',
                                        'use_datadrift': use_datadrift,
                                        'type_dataset': type_dataset},
                                  create_new_version=True)
        except Exception as ex:
            print(ex)

    def __frequency_tremor(self, df):
        axis = ["accZ_mean", "accX_mean"]
        for axi in axis:
            sns.set_style('whitegrid')
            plt.rcParams['font.family'] = 'Dejavu Sans'
            plt.figure(figsize=(16,8))

            sns.set_palette("Set1", desat=0.80)
            facetgrid = sns.FacetGrid(df, hue='Tremor', size=6,aspect=2)
            facetgrid.map(sns.distplot, f"{axi}", hist=False)\
                .add_legend()
            self.__run.log_image(f"Parkinson Tremor - {axi}", plot=plt)
    
    def __tremor_acceleration_energy(self, df):
        plt.figure(figsize=(6,8))
        sns.boxplot(x='Tremor', y='accX_energy',data=df, showfliers=False, saturation=1)
        plt.ylabel('Acceleration Energy X')
        self.__run.log_image(f"Parkinson Tremor - Acceleration Energy X", plot=plt)

    def __perform_tsne(self, X_data, y_data, perplexities, n_iter=1000, img_name_prefix='t-sne'):
        for index,perplexity in enumerate(perplexities):
            print('\nperforming tsne with perplexity {} and with {} iterations at max'.format(perplexity, n_iter))
            X_reduced = TSNE(verbose=2, perplexity=perplexity).fit_transform(X_data)
            print('Done..')
            
            print('Creating plot for this t-sne visualization..')
            df = pd.DataFrame({'x':X_reduced[:,0], 'y':X_reduced[:,1] ,'label':y_data})
            
            sns.lmplot(data=df, x='x', y='y', hue='label', fit_reg=False, size=8,\
                    palette="Set1",markers=['*', 'o'])
            plt.title("perplexity : {} and max_iter : {}".format(perplexity, n_iter))
            img_name = img_name_prefix + '_perp_{}_iter_{}.png'.format(perplexity, n_iter)
            print('saving this plot as image in present working directory...')
            self.__run.log_image(f"Parkinson Tremor - {img_name}", plot=plt)
    
    def __execute_tsne(self, df):
        X_norm= normalize(df.drop(['Tremor'], axis=1), norm='l2')
        X_new2 = MinMaxScaler().fit_transform(X_norm)
        X_pre_tsne = X_new2
        y_pre_tsne =df['Tremor']
        self.__perform_tsne(X_data = X_pre_tsne,y_data=y_pre_tsne, perplexities =[2,5,10,20,50])

if __name__ == '__main__':
    analysis = ExploratoryAnalysis()
    analysis.main()
