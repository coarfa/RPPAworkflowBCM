#!/usr/bin/env python
import xlsxwriter
import xlrd
import pandas as pd
# workaround for cluster's error: "    raise RuntimeError('Invalid DISPLAY variable')
# RuntimeError: Invalid DISPLAY variable"
import matplotlib
matplotlib.use('Agg')
# Don't remove this two lines otherwise it's not going to work on the cluster
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import argparse
import sys
import os
import numpy as np
# workaround for cluster's ero r: "ImportError: /share/apps/anaconda/2.5.0/lib/libreadline.so.6: undefined symbol: PC"
# don't remove this line otherwise it'll not work on the cluster
import readline
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
import csv
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
stats1 = importr('stats')
# Allows adding newline to the argparse description
from argparse import RawTextHelpFormatter
from collections import OrderedDict
import re
import copy
import imp
# Importing module commonFunctions
common_functions = imp.load_source("commonFunctions", os.path.abspath(os.path.join(os.path.dirname(__file__),  '../..',
                                   'commonFunctions.py')))

# module load R/3.1.1
# module load anaconda/2.5.0

class rppaStep1:

    """
    DESCRIPTION: This script executes the first steps of the metabolomics analysis.
    This script:
      - Checks for input consistency: number of groups and samples, repeated data
      - Selects the internal standard with the lower coefficient of variation
      - Plots the distribution for the selected internal standard
      - Normalizes the data based on the internal standards or using Interquartile Range (IQR)
      - Creates a spreadsheet with the normalized data
      - Plots liver control boxplots
      - Plots boxplot comparing the data for each method
      - Plots boxplot comparing the data for a certain method and comparison (optional)
      - Performs comparisons between defined groups (t-test or anova) and generates a report with p-value,
      adjusted p-value, and fold change (for t-test)
      - Calculates the z-scores for each comparison between defined groups and calls createHeatmap.py in order to
      generate a heatmap (just for values <= FDR)
      - Calculates the z-scores of the normalized data and and calls createHeatmap.py in order to
      generate a heatmap
    ARGUMENTS:
        --input | -i => JSON file containing the input and parameters of the project
    USAGE:
        python rppaStep1.py -i JSON_file
    USAGE EXAMPLE:
        python rppaStep1.py -i ../../JSON_input.json
    NOTES:
        1: On the cluster, in order to avoid the error "assert 0 <= colx < X12_MAX_COLS \nAssertionError", run:
        python -O rppaStep1.py -i ../../JSON_input.json

        2: When executing this script, if you get the error "assert 0 <= colx < X12_MAX_COLS \nAssertionError"
        and the fix above doesn't work do:

        Open the xlsx file and save as a new name and run the script again. There is a bug with the xlrd and this
        happens some times. I can't reproduce the error though.

    """

    heatmap_script = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
    heatmap_script = os.path.abspath(heatmap_script + '/createHeatmap.py')

    pca_script = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
    pca_script = os.path.abspath(pca_script + '/createPCA.py')

    # JSON Project variables
    project_name = None
    output_dir = None
    excel_input = None
    phenotype_input = None
    # BCM, MDA, TCGA
    input_format = 'BCM'
    p_value = None
    correction = None
    fold_change = None
    fold_change_color_min = -1
    fold_change_color_max = 1
    max_intensity = None
    technical_replication = False
    # if tuple with 2 groups
    group_comparisons = []
    user_group_color = None
    use_user_group_color = False
    # Norm sheet name
    norm_sheet_name = None
    # Mouse norm sheet name
    norm_sheet_name1 = None
    scale_heatmap = 1
    group_palette = None
    number_of_groups = None
    summary_prefix = None
    bio_rep_prefix = None
    cv_sheet_name = None
    t_test_prefix = None
    anova_prefix = None
    z_score_prefix = None
    nan = None
    header_background_1 = None
    header_background_2 = None
    fold_change_prefix = None
    p_value_prefix = None
    max_intensity_prefix = None
    adj_p_value_prefix = None
    signature_prefix = None
    signature_combined_prefix = None
    filter_prefix = '_filter'
    pathway_files = None

    # heatmap variables
    color_map_extension = None
    color_map_distance = None
    create_color_map = False
    distance_metrics = None
    heatmap_extension = None
    color_gradient = None

    # PCA plot variables
    angles = [0, 30, 60, 90]
    pca_figure_size = None
    pca_prefix = 'pca_3d_plot_'
    pca_extension = '.png'
    pca_resolution = 600
    use_group_palette_for_pca = False

    # INTERNAL VARIABLES
    norm_data_frame = None
    pheno_data_frame = None
    # Create a dictionary with treatment group and dataframe for each group
    grouped_data_dict = {}
    # sample dictionary
    sample_group = {}
    # Max intensity dictionary
    max_intensity_dict = {}
    workbook_summary = None
    anova_workbook_summary = None
    format = None
    anova_format = None
    anova_data_frame = None
    t_test_data_frame = None
    comparisons_results = {}
    header_dict = {}
    index_header = 'AB_ID'
    t_test_filter_list = []
    anova_filter_list = []
    new_sheet_list = []
    sig_sheet_list = []
    # Dictionary that makes a gene to a list of antibodies
    gene_to_ab_dict = defaultdict(list)
    # Dictionary that links the signature long name to an alias like sig_1, ..., sig_n in order to
    # decrease the size of the sheet names
    signature_code = {}
    sig_count = 1
    COMP_INFIX = '.over.'

    def __init__(self, args):
        self.set_parameters(args)

    @staticmethod
    def get_parameters():
        parser = argparse.ArgumentParser(description='This script executes the first steps of the RPPA analysis.\n'
                                            " This script:\n"

                                            '- Creates a CSV with the z-scores\n', formatter_class=RawTextHelpFormatter)
        parser._optionals.title = "PARAMETERS"
        parser.add_argument('-i', '--input', type=str, required=True,
                            help='JSON file containing the input and parameters of the project')
        if len(sys.argv[1:]) == 0:
            parser.print_help()

        try:
            args = parser.parse_args()
        except:
            args = None
        return args

    # Set input parameters
    def set_parameters(self, args):
        print '\n\n################################'
        print '# Running ' + os.path.basename(__file__) + ' #'
        print '################################\n'
        print '==> Setting parameters\n'
        self.args = args
        # setting input file name
        self.input = os.path.abspath(self.args.input)
        parameters = ('## PARAMETERS ## \n' + 'INPUT: ' + self.input + '\n')
        print parameters

    # Set variables based on the data in the JSON file
    def set_input(self):
        print '==> Setting input'
        json_data = common_functions.read_JSON(self.input)
        # Setting project variables
        self.project_name = json_data["project"][0]["project_name"]
        self.output_dir = json_data["project"][0]["output_dir"]
        if not os.path.exists(self.output_dir):
            print 'Creating output directory ' + self.output_dir
            os.makedirs(self.output_dir)
        self.excel_input = os.path.abspath(json_data["project"][0]["excel_input"])
        self.phenotype_input = os.path.abspath(json_data["project"][0]["phenotype_input"])
        if "input_format" in json_data["project"][0]:
            self.input_format = json_data["project"][0]["input_format"]
        self.p_value = json_data["project"][0]["p_value"]
        self.correction = json_data["project"][0]["correction"]
        self.fold_change = json_data["project"][0]["fold_change"]
        if "fold_change_color_min" in json_data["project"][0]:
            self.fold_change_color_min = json_data["project"][0]["fold_change_color_min"]
        if "fold_change_color_max" in json_data["project"][0]:
            self.fold_change_color_max = json_data["project"][0]["fold_change_color_max"]
        self.max_intensity = json_data["project"][0]["max_intensity"]
        if "technical_replication" in json_data["project"][0] \
                and json_data["project"][0]["technical_replication"] == 'True':
            self.technical_replication = True
        # Creating group comparison list
        for comparison in json_data["project"][0]["group_comparisons"]:
            self.group_comparisons.append(comparison)

        self.norm_sheet_name = json_data["project"][0]["norm_sheet_name"]
        self.norm_sheet_name1 = json_data["project"][0]["norm_sheet_name1"]
        if "scale_heatmap" in json_data["project"][0]:
            self.scale_heatmap = json_data["project"][0]["scale_heatmap"]
        self.number_of_groups = json_data["project"][0]["number_of_groups"]
        self.group_palette = json_data["project"][0]["group_palette"]
        if "pathway_files" in json_data["project"][0]:
            self.pathway_files = json_data["project"][0]["pathway_files"]

        # Creating a list with the filter parameters
        # If more parameters were added to the comparison they will be put together
        # This list is just to make easier to make a list for each comparison
        self.t_test_filter_list.append({'p-value': self.p_value, 'intensity': self.max_intensity, 'fold_change': self.fold_change})
        self.anova_filter_list.append(self.p_value)

        print '\nProject:\n----------------'
        print 'Project name: ' + self.project_name
        print 'Output dir: ' + self.output_dir
        print 'Excel input: ' + self.excel_input
        print 'Phenotype input: ' + str(self.phenotype_input)
        print 'p-value: ' + str(self.p_value)
        print 'p-value correction: ' + str(self.correction)
        print 'Fold change: ' + str(self.fold_change)
        print 'Fold change color min: ' + str(self.fold_change_color_min)
        print 'Fold change color max: ' + str(self.fold_change_color_max)
        print 'Maximum intensity criteria: ' + str(self.max_intensity)
        print 'Technical replication: ' + str(self.technical_replication)
        print 'Group comparisons: ' + str(self.group_comparisons)
        print 'Norm sheet name: ' + self.norm_sheet_name
        print 'Norm sheet name (in general Mouse_Norm): ' + self.norm_sheet_name1
        print 'Scale heatmap in excel file: ' + str(self.scale_heatmap)
        print 'Number of groups: ' + str(self.number_of_groups)
        print 'Group palette: ' + self.group_palette
        print 'Pathway files: ' + str(self.pathway_files)

        # Setting configuration variables
        self.summary_prefix = json_data["configuration"][0]["summary_prefix"]
        self.bio_rep_prefix = json_data["configuration"][0]["bio_rep_prefix"]
        self.cv_sheet_name = json_data["configuration"][0]["cv_sheet_name"]
        self.t_test_prefix = json_data["configuration"][0]["t_test_prefix"]
        self.anova_prefix = json_data["configuration"][0]["anova_prefix"]
        self.nan = json_data["configuration"][0]["not_a_number"]
        self.header_background_1 = json_data["configuration"][0]["header_background_1"]
        self.header_background_2 = json_data["configuration"][0]["header_background_2"]
        self.fold_change_prefix = json_data["configuration"][0]["fold_change_prefix"]
        self.p_value_prefix = json_data["configuration"][0]["p_value_prefix"]
        self.max_intensity_prefix = json_data["configuration"][0]["max_intensity_prefix"]
        self.adj_p_value_prefix = json_data["configuration"][0]["adj_p_value_prefix"]
        self.signature_prefix = json_data["configuration"][0]["signature_prefix"]
        self.signature_combined_prefix = json_data["configuration"][0]["signature_combined_prefix"]
        self.z_score_prefix = json_data["configuration"][0]["z_score_prefix"]

        print '\nConfiguration:\n-----------------'
        print 'Summary prefix: ' + self.summary_prefix
        print 'Biological replicates prefix: ' + self.bio_rep_prefix
        print 'CV sheet name: ' + self.cv_sheet_name
        print 't-test prefix: ' + self.t_test_prefix
        print 'ANOVA prefix: ' + self.anova_prefix
        print 'Not a number: ' + self.nan
        print 'Header background 1: ' + self.header_background_1
        print 'Header background 2: ' + self.header_background_2
        print 'p-value prefix: ' + self.p_value_prefix
        print 'Fold change prefix: ' + self.fold_change_prefix
        print 'Max intensity prefix: ' + self.max_intensity_prefix
        print 'Adjusted p-value prefix: ' + self.adj_p_value_prefix
        print 'Signature prefix: ' + self.signature_prefix
        print 'Signature combined prefix: ' + self.signature_combined_prefix
        print 'z-score prefix: ' + self.z_score_prefix

        # Heatmap variables
        self.heatmap_extension = json_data["configuration"][0]["heatmap"][0]["heatmap_extension"]
        self.distance_metrics = json_data["project"][0]["heatmap"][0]["distance_metrics"]
        if "color_map_distance" in json_data["configuration"][0]["heatmap"][0]:
            self.color_map_distance = json_data["configuration"][0]["heatmap"][0]["color_map_distance"]
        else:
            self.color_map_distance = self.distance_metrics[0]
        if "create_color_map" in json_data["configuration"][0]["heatmap"][0] and \
                        json_data["configuration"][0]["heatmap"][0]["create_color_map"] == "True":
            self.create_color_map = True
        self.color_map_extension = json_data["configuration"][0]["heatmap"][0]["color_map_extension"]
        self.color_gradient = json_data["project"][0]["heatmap"][0]["color_gradient"]

        print '\nHeatmap variables:\n-----------------'
        print 'Heatmap extension: ' + self.heatmap_extension
        print 'Create color map: ' + str(self.create_color_map)
        print 'Color map extension: ' + str(self.color_map_extension)
        print 'Distance metrics: ' + str(self.distance_metrics)
        print 'Color map distance: ' + str(self.color_map_distance)
        print 'Color gradient: ' + str(self.color_gradient)

        # PCA plot variables
        if "pca_pca_prefix" in json_data["configuration"][0]["pca"][0]:
            self.pca_prefix = json_data["configuration"][0]["pca"][0]["pca_prefix"]
        self.pca_figure_size = tuple(json_data["configuration"][0]["pca"][0]["pca_figure_size"])
        if "pca_extension" in json_data["configuration"][0]["pca"][0]:
            self.pca_extension = json_data["configuration"][0]["pca"][0]["pca_extension"]
        if "pca_resolution" in json_data["configuration"][0]["pca"][0]:
            self.pca_resolution = json_data["configuration"][0]["pca"][0]["pca_resolution"]
        if "use_group_palette" in json_data["configuration"][0]["pca"][0] \
                and json_data["configuration"][0]["pca"][0]["use_group_palette"] == "True":
            self.use_group_palette_for_pca = True

    def run_step1(self):
        print '\n==>Starting run_step1\n'
        # Parsing phenotype input anf creating dataframe
        if self.phenotype_input is not None:
            self.pheno_data_frame = pd.read_table(self.phenotype_input, sep="\t", index_col=0, converters={'treat': str})
            if self.input_format is not None and (self.input_format == 'MDA'or self.input_format == 'TCGA'):
                self.exec_analysis_mda_tcga()
            else:
                self.exec_analysis_baylor()

    def prepare_xlsx_data(self):
        print '==> Preparing xslx data'
        workbook = xlrd.open_workbook(self.excel_input)
        # Parsing xslx input and creating norm dataframe
        # Checking if sheet exists
        if workbook.sheet_by_name(self.norm_sheet_name):
            print '===> Sheet ' + self.norm_sheet_name + ' found'
            # The xlsx has two headers
            # first headers is the header of the dataframe
            # second header is going to be inside the dataframe like data
            # but it's going to be treated differently
            df = pd.read_excel(self.excel_input, sheetname=self.norm_sheet_name, index_col=0)
        else:
            sys.exit('Sheet Norm ' + self.norm_sheet_name + " wasn't found. Please check your xlsx input")
        # Test if Mouse_Norm exists
        if self.norm_sheet_name1 in workbook.sheet_names():
            print '===> Sheet ' + self.norm_sheet_name1 + ' found'
            df1 = pd.read_excel(self.excel_input, sheetname=self.norm_sheet_name1, index_col=0)
            print '===> Concatenating dataframes'
            # Concatenate data old dataframe with new
            # df1.iloc[1:, 0:] skips the 2nd header in the xlsx.
            # The first header is the header of the data
            frames = [df, df1.iloc[1:, 0:]]
            self.norm_data_frame = pd.concat(frames)

        else:
            print 'Sheet ' + self.norm_sheet_name1 + " wasn't found."
            print 'Just ' + self.norm_sheet_name + ' will be used'
            self.norm_data_frame = df

        # Create a data frame just with bio replicates values
        new_column_header = []
        # Get values from column F in xlsx file
        # Make the values float
        data = self.norm_data_frame.iloc[1:, 4:].astype(np.float64)

        # Make first header (header is Multilevel) the data frame header
        # Remove \.\d from the end of each field of the header

        for i in range(0, len(data.columns.get_level_values(0))):
            if not self.technical_replication:
                clean_header = re.split(r"(\..$)", data.columns.get_level_values(0)[i])[0]
            else:
                clean_header = data.columns.get_level_values(0)[i]
            new_column_header.append(clean_header)
            # Create header dictionary in order to be able to access the link between the two headers
            # + 4 because to skip AB_ID. ... Swiss_ID and go to rep
            self.header_dict[clean_header] = self.norm_data_frame.iloc[0, i + 4]

        # Group data using the new_column_header
        group_bio_rep = data.groupby([new_column_header], axis=1, sort=False)

        # Create data frame with the median of each group
        bio_rep_median = group_bio_rep.aggregate(np.median)


        bio_rep_cv = data.groupby([new_column_header], axis=1, sort=False).std()/data.groupby([new_column_header],
                                                                                              axis=1, sort=False).mean()



        # Group 2nd header data using the new_column_header
        group_column_header = (self.norm_data_frame.iloc[0:1, 4:]).groupby([new_column_header], axis=1, sort=False)
        # Create a data frame with new_column_new as header and 2nd header as data
        bio_rep_names = group_column_header.first()


        # CREATE MEDIAN DATA FRAME + FILE
        # Create a data frame containing the median data together with the additional data
        # in self.norm_data_frame[:, 0:4]
        bio_rep_data_frame = (self.norm_data_frame.ix[:, 0:4]).join(bio_rep_names.append(bio_rep_median))

        bio_rep_data_frame_cv = (self.norm_data_frame.ix[:, 0:4]).join(bio_rep_names.append(bio_rep_cv))

        # If technical replicates exist
        # Use whole data
        if self.technical_replication:
            print 'Using technical replication'
            input_data = data
            input_data_names = self.norm_data_frame.iloc[0:1, 4:]
        # Use median data
        else:
            print 'Not using technical replication'
            input_data = bio_rep_median
            input_data_names = bio_rep_names

        self.group_data(input_data)

        return bio_rep_data_frame, bio_rep_data_frame_cv

    def get_background_and_font_color(self, v):
        # Setting color gradient
        color_gradients = common_functions.create_custom_color_gradient()
        cmap = matplotlib.cm.get_cmap(color_gradients[self.color_gradient])
        norm = matplotlib.colors.Normalize(vmin=self.fold_change_color_min, vmax=self.fold_change_color_max)
        rgba = cmap(norm(v))
        hex = common_functions.rgb_to_hex((rgba[0], rgba[1], rgba[2]))
        font = common_functions.get_font_color([rgba[0], rgba[1], rgba[2]])
        return hex, font

    def create_pca_plot(self, bio_rep_data_frame):
        print common_functions.func_start_header(self.create_pca_plot.__name__)
        # Get the just the numbers from the dataframe
        if self.input_format == 'MDA':
            df = bio_rep_data_frame.iloc[1:, 6:]
        elif self.input_format == 'TCGA':
            df = bio_rep_data_frame.iloc[1:, 1:]
        else:
            df = bio_rep_data_frame.iloc[1:, 4:]
        new_dict = {}
        for key in self.grouped_data_dict:
            for c in self.grouped_data_dict[key].columns:
                new_dict[c] = key

        new_header = [(new_dict[c], c) for c in df.columns]
        df.columns = pd.MultiIndex.from_tuples(new_header)

        pca_txt_file = os.path.join(self.output_dir, self.pca_prefix + '.txt')
        df.to_csv(pca_txt_file)

        print '==> Starting pca for'
        cmd = 'python ' + self.pca_script + ' -i ' + self.input + ' -c ' + pca_txt_file
        print '==> Ready to execute: ' + cmd + '\n'
        stderr, result = common_functions.exec_command(cmd)

        if result == 0:
            print common_functions.bcolors.OKGREEN + 'PCA created without errors. See log bellow:' + common_functions.bcolors.ENDC
            print stderr
            print common_functions.bcolors.OKGREEN + '------ END OF PCA LOG -----' + common_functions.bcolors.ENDC
        else:
            print common_functions.bcolors.FAIL + 'Error creating PCA'
            print stderr
            print common_functions.bcolors.FAIL + '------ END PCA ERROR LOG -----' + common_functions.bcolors.ENDC



        # print '==> Create PCA'
        # # Create PCD of Bio rep
        # fig = plt.figure(1, figsize=(4, 3))
        # plt.clf()
        # ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        # plt.cla()



        # # Remove NANs
        # df = df.dropna()
        # # Transpose dara to do the PCA by groups (phenotypes)
        # df = df.T

        # Calculating the PCA
        # x_reduced = PCA(n_components=3).fit_transform(df)
        # pca_df = pd.DataFrame(x_reduced, index=df.index.values)
        #
        # if self.use_group_palette_for_pca:
        #     colors = common_functions.set_group_color(self.grouped_data_dict.keys(),
        #                                               len(self.grouped_data_dict.keys()),
        #                                               self.group_palette)
        # for ang in self.angles:
        #     fig = plt.figure(1, figsize=self.pca_figure_size)
        #     ax = Axes3D(fig, elev=30, azim=ang)
        #
        #     for key in self.grouped_data_dict:
        #         # Get the slice of the pca_df where the samples belong to the group (key)
        #         slice = pca_df.ix[self.grouped_data_dict[key].columns]
        #
        #         if self.use_group_palette_for_pca:
        #             ax.scatter(slice[0], slice[1], slice[2], label=key, color=colors[key], s=40, depthshade=False)
        #         else:
        #             ax.scatter(slice[0], slice[1], slice[2], label=key, s=40, depthshade=False)
        #         for i in slice.index:
        #             ax.text3D(slice.ix[i, 0], slice.ix[i, 1], slice.ix[i, 2], i, size=2.5)
        #     ax.tick_params(labelsize=5)
        #     ax.w_xaxis.set_label([])
        #
        #     plt.legend(loc='best', fontsize=6)
        #     #print self.output_dir + '/' + self.pca_prefix + str(ang) + self.pca_extension
        #     plt.savefig(self.output_dir + '/' + self.pca_prefix + str(ang) + self.pca_extension, bbox_inches='tight',
        #                 pad_inches=0, dpi=self.pca_resolution)

    def exec_analysis_baylor(self):
        # Adding the first elements to the new_sheet_list
        # This list will be used to sort the final spreadsheet
        self.new_sheet_list.append(self.summary_prefix)
        self.new_sheet_list.append(self.bio_rep_prefix)
        self.new_sheet_list.append(self.cv_sheet_name)

        print '==> Executing Analysis '
        outfile = self.output_dir + '/' + self.summary_prefix + '.xlsx'
        self.workbook_summary = xlsxwriter.Workbook(outfile)
        print '===> Creating summary output: ' + outfile
        # Set formats for workbook summary
        self.format = common_functions.set_formats(self.workbook_summary)


        # BIO REP
        (bio_rep_data_frame, bio_rep_data_frame_cv) = self.prepare_xlsx_data()
        self.write_bio_rep(bio_rep_data_frame, self.bio_rep_prefix, create_xlsx=True)
        # Create PCA plot

        self.create_pca_plot(bio_rep_data_frame)

        self.write_bio_rep(bio_rep_data_frame_cv, self.cv_sheet_name, special_format='percent')

        # EXEC COMPARISONS
        (signature_combined, signature_dict) = self.exec_comparisons()
        self.write_comparisons()
        self.write_signature_combined(signature_combined)
        self.write_summary_sheet(signature_dict)

        # Adding the signatures to the new_sheet_list to sort the sheet in the spreadsheet
        self.new_sheet_list += self.sig_sheet_list
        # Sorting the workbook according to the order in the new_sheet_list
        self.workbook_summary.worksheets_objs.sort(key=lambda x: self.new_sheet_list.index(x.name))

        self.workbook_summary.close()

    def exec_analysis_mda_tcga(self):
        # Adding the first elements to the new_sheet_list
        # This list will be used to sort the final spreadsheet
        self.new_sheet_list.append(self.summary_prefix)
        self.new_sheet_list.append(self.bio_rep_prefix)

        print '==> Executing Analysis '
        outfile = self.output_dir + '/' + self.summary_prefix + '.xlsx'
        self.workbook_summary = xlsxwriter.Workbook(outfile)
        print '===> Creating summary output: ' + outfile
        # Set formats for workbook summary
        self.format = common_functions.set_formats(self.workbook_summary)
        n = None

        # BIO REP
        if self.input_format == 'MDA':
            n = 6
        elif self.input_format == 'TCGA':
            n = 1
        else:
            sys.exit(common_functions.bcolors.FAIL + 'Input format unkown: ' + self.input_format)
        # n is the number of column - 1 where the data starts from 0
        # n = 6 for MDA
        # n = 1 for TCGA
        bio_rep_data_frame = self.prepare_xlsx_data_mda_tcga(n)
        # Create PCA plot

        self.create_pca_plot(bio_rep_data_frame)
        self.write_bio_rep(bio_rep_data_frame, self.bio_rep_prefix, create_xlsx=True)

        # EXEC COMPARISONS
        (signature_combined, signature_dict) = self.exec_comparisons_mda_tcga()
        self.write_comparisons()
        self.write_signature_combined(signature_combined)

        self.write_summary_sheet(signature_dict)

        # Adding the signatures to the new_sheet_list to sort the sheet in the spreadsheet
        self.new_sheet_list += self.sig_sheet_list
        # Sorting the workbook according to the order in the new_sheet_list
        self.workbook_summary.worksheets_objs.sort(key=lambda x: self.new_sheet_list.index(x.name))

        self.workbook_summary.close()

    def prepare_xlsx_data_mda_tcga(self, n):
        # n is the number of column - 1 where the data starts from 0
        # n = 6 for MDA
        # n = 1 for TCGA
        print '==> Preparing xslx data'
        workbook = xlrd.open_workbook(self.excel_input)
        df = pd.read_excel(self.excel_input, sheetname='Sheet1', index_col=0)
        self.norm_data_frame = df

        # Create a data frame just with bio replicates values
        new_column_header = []

        # Get values from column F in xlsx file
        # Make the values float
        data = df.iloc[0:, n:].astype(np.float64)

        self.group_data(data)

        return self.norm_data_frame

    # Create bio rep sheet
    # Create bio rep .xlsx
    def write_bio_rep(self, df, sheet_name, create_xlsx=False, special_format=None):

        # Last number from 0 of descrition column for BCM data
        last_desc_column = 3
        # Start range for last loop
        # BCM data has two headers and the second is in the first row, so we need to start from the 2nd row (1)
        start = 1
        if self.input_format == 'MDA':
            # Last number from 0 of description column for MDAnderson data
            last_desc_column = 5
            # Start range for last loop
            # MDAnderson data has just one header, so we can start from the first row (0)
            start = 0
        elif self.input_format == 'TCGA':
            # Last number from 0 of description column for TCGA data
            last_desc_column = 0
            # Start range for last loop
            # MDAnderson data has just one header, so we can start from the first row (0)
            start = 0

        # Write to summary
        print '==> Writing bio rep data to summary'
        worksheet = self.workbook_summary.add_worksheet(sheet_name)
        # Write indexes
        for i in range(0, len(df.index)):
            if i == 0 and self.input_format != 'MDA':
                worksheet.write(i + 1, 0, df.index[i], self.format['1'])
            else:
                worksheet.write(i + 1, 0, df.index[i], self.format['bold'])

        # Write header
        if self.input_format == 'MDA':
            # Write column header
            for c in range(0, len(df.columns)):
                if c < last_desc_column:
                    worksheet.write(0, c + 1, df.columns[c], self.format['1'])
                # Add right border
                elif c == last_desc_column:
                    worksheet.write(0, c + 1, df.columns[c], self.format['4'])
                else:
                    worksheet.write(0, c + 1, df.columns[c], self.format['8'])
        else:
            # Write column header 1 and 2
            for c in range(0, len(df.columns)):
                if c < last_desc_column:
                    worksheet.write(1, c + 1, df.iloc[0, c], self.format['1'])
                # Add right border
                elif c == last_desc_column:
                    worksheet.write(0, c + 1, '', self.format['3'])
                    worksheet.write(1, c + 1, df.iloc[0, c], self.format['4'])
                else:
                    worksheet.write(0, c + 1, df.columns[c], self.format['7'])
                    worksheet.write(1, c + 1, df.iloc[0, c], self.format['8'])

        #for r in range(start, len(df.index) - 1):
        for r in range(start, len(df.index)):
            for c in range(0, len(df.columns)):
                # Add right border
                if c == last_desc_column:
                    if pd.isnull(df.iloc[r, c]):
                        worksheet.write_blank(r + 1, c + 1, self.nan)
                    else:
                        worksheet.write(r + 1, c + 1, df.iloc[r, c], self.format['5'])
                elif c < last_desc_column:
                    worksheet.write(r + 1, c + 1, str(df.iloc[r, c]))
                # No border
                else:
                    if pd.isnull(df.iloc[r, c]):
                        worksheet.write_blank(r + 1, c + 1, None, self.format['9'])
                    else:
                        if special_format == 'percent':
                            value = df.iloc[r, c]
                            if  df.iloc[r, c] * 100 >= 20:
                                print value
                                worksheet.write(r + 1, c + 1, value, self.format['9_percent_color'])
                            else:
                                worksheet.write(r + 1, c + 1, value, self.format['9_percent'])
                        else:
                            worksheet.write(r + 1, c + 1, df.iloc[r, c], self.format['9'])
                    # Increase the column width
                    worksheet.set_column(c + 1, c + 1, 12)

        if create_xlsx:
            # Write data to xlsx
            print '==> Writing bio rep data to xlsx'
            df.to_excel(os.path.join(self.output_dir, self.bio_rep_prefix + '.xlsx'), na_rep=self.nan)
            # to_csv(self.output_dir + '/' + self.bio_rep_prefix + '.csv', sep="\t", na_rep=self.nan, index=True)

    @staticmethod
    def calculate_max_intensity(df):
        max_intensity = {}
        for i in df.index:
            # max intensity is the maximum value of the row in the data frame
            max_intensity[i] = np.nanmax((df.ix[i, :]).values)
        return pd.Series(max_intensity)

    # Separates the data in phenotype groups
    def group_data(self, in_df):
        print '==> Grouping data according to phenotype'
        # If pheno_data_frame has data
        if not self.pheno_data_frame.empty:
            treatment_group = []

            for i in range(0, len(self.pheno_data_frame.index)):
                treatment_group.append(self.pheno_data_frame.ix[i, "treat"])
                if self.input_format == 'MDA':
                    aux = self.pheno_data_frame.index[i]
                    aux = aux.replace(' ', '_')
                    aux = common_functions.replace_chars(aux)
                    self.sample_group[aux] = self.pheno_data_frame.ix[i, "treat"]
                else:
                    self.sample_group[self.pheno_data_frame.index[i]] = self.pheno_data_frame.ix[i, "treat"]

            group_data_frame = in_df.groupby([treatment_group], axis=1, sort=False)

            # Create a dictionary with treatment group and dataframe for each group
            for keys in group_data_frame.groups:
                self.grouped_data_dict[keys] = group_data_frame.get_group(keys)
            # Check if the number of groups is the same in the JSON file and in the xlsx
            # if it's not might have problems during heatmap creation
            if len(self.grouped_data_dict) != self.number_of_groups:
                sys.exit(common_functions.bcolors.FAIL + 'The number of groups declared in the JSON file ('
                         + str(
                    self.number_of_groups) + ') is different from the number of groups found in the xlsx file ('
                         + str(len(self.grouped_data_dict)) + ').')
        else:
            sys.exit('No phenotype data. Please check file ' + self.phenotype_input + ' content.')

    def check_for_zero(self, df):
        if 0 in df:
            sys.exit(common_functions.bcolors.FAIL + "Error: Zero found in the data. Impossible to calculate log2."
                     + ' Please replace the value or delete the column and rerun the script.')

    # Execute comparisons: t-test or ANOVA
    def exec_comparisons(self):
        signatures_combined = None
        signature_dict = OrderedDict()
        # Comparisons
        # If group_comparisons not empty
        if self.group_comparisons:
            print '==> Starting comparisons: '

            for i in range(0, len(self.group_comparisons)):

                # Execute t-test
                if not self.group_comparisons[i].get("stats") or \
                        (self.group_comparisons[i].get("stats") and self.group_comparisons[i]["stats"] == "t-test"):
                    # Normalize data
                    # Read controls. Calculate log2
                    for c in range(0, len(self.group_comparisons[i]["control"])):
                        self.check_for_zero(self.grouped_data_dict[self.group_comparisons[i]["control"][c]])
                        if c == 0:
                            control_data = self.grouped_data_dict[self.group_comparisons[i]["control"][c]]
                            control_data_log2 = np.log2(self.grouped_data_dict[self.group_comparisons[i]["control"][c]])
                            control_names = self.group_comparisons[i]["control"][c]
                        else:
                            # If more than one control add data frame log2 to control_data
                            control_data = control_data.join(self.grouped_data_dict[self.group_comparisons[i]["control"][c]])
                            control_data_log2 = control_data_log2.join(
                                np.log2(self.grouped_data_dict[self.group_comparisons[i]["control"][c]]))
                            control_names += '_' + self.group_comparisons[i]["control"][c]
                    # Read treatments. Calculate log2
                    for t in range(0, len(self.group_comparisons[i]["test"])):
                        self.check_for_zero(self.grouped_data_dict[self.group_comparisons[i]["test"][t]])
                        if t == 0:
                            treatment_data = self.grouped_data_dict[self.group_comparisons[i]["test"][t]]
                            treatment_data_log2 = np.log2(self.grouped_data_dict[self.group_comparisons[i]["test"][t]])
                            treatment_names = self.group_comparisons[i]["test"][t]
                        else:
                            # If more than one control add data frame log2 to control_data
                            treatment_data = treatment_data.join(self.grouped_data_dict[self.group_comparisons[i]["test"][t]])
                            treatment_data_log2 = treatment_data_log2.join(
                                np.log2(self.grouped_data_dict[self.group_comparisons[i]["test"][t]]))
                            treatment_names += '_' + self.group_comparisons[i]["test"][t]

                    # Creating comparison name. Test come first.
                    comparison_name = treatment_names + '.over.' + control_names
                    print '===> Comparison: ' + comparison_name
                    # Execute t-test
                    (p_value, fold_change) = self.t_test(control_data_log2, treatment_data_log2)
                    # Adjust p-value
                    p_value_adjusted = self.adjust_p_value(p_value, self.correction)
                    # Max intensity for comparison
                    # This result will be used on signatures
                    max_intensity = self.calculate_max_intensity(pd.concat([control_data, treatment_data], axis=1))

                    # Creating a list of parameters in order to be able to work with signatures
                    # for one or more filters
                    # Since in python setting a variable actually sets a reference to the variable.
                    # We need to simply copy the list to the new list:
                    filter_aux = self.t_test_filter_list[:]
                    if self.group_comparisons[i].get("filter"):
                        for f in self.group_comparisons[i]['filter']:
                            filter_aux.append(f)
                    # Creating data frame with p-value, fold change and ajusted p-value
                    label1 = self.p_value_prefix + '_' + comparison_name
                    label2 = self.adj_p_value_prefix + '_' + comparison_name
                    label3 = self.fold_change_prefix + '_' + comparison_name
                    label4 = self.max_intensity_prefix + '_' + comparison_name
                    df_aux = pd.DataFrame({label1: p_value,
                                           label2: p_value_adjusted,
                                           label3: fold_change,
                                           label4: max_intensity},
                                          index=control_data.index, columns=[label1, label2, label3, label4])
                    if 't-test' in self.comparisons_results:
                        self.comparisons_results[self.t_test_prefix][comparison_name] = df_aux
                    else:
                        self.comparisons_results[self.t_test_prefix] = {}
                        self.comparisons_results[self.t_test_prefix][comparison_name] = df_aux

                    c = 1
                    for f in filter_aux:
                        # If any key is missing in the filter it's going to be replaced by the default value
                        if 'p-value' not in f:
                            self.p_value
                        else:
                            p_v = f['p-value']
                        if 'fold_change' not in f:
                            f_c = self.fold_change
                        else:
                            f_c = f['fold_change']
                        if 'intensity' not in f:
                            m_e = self.max_intensity
                        else:
                            m_e = f['intensity']

                        # Adding suffix to label in order to differ comparison by filter
                        suffix = '_' + str(p_v) + '-' + str(f_c) + '-' + str(m_e)
                        l1 = label1 + suffix
                        l2 = label2 + suffix
                        l3 = label3 + suffix
                        l4 = label4 + suffix

                        new_column = [l1, l2, l3, l4]
                        # df_aux.copy aims to avoid making changes in the copy
                        # Since python sets a reference to the variable instead of setting the variable
                        # Doing this when I change the name of the columns bellow it doesn't affect the
                        # data stored in self.comparisons_results (df_aux)
                        df_sig = df_aux.copy()
                        df_sig.columns = new_column

                        # Find signature for dataframe
                        signature = self.find_signatures(df_sig, p_v, f_c, m_e)

                        # code is used in order to identify the comparison and its parameters without creating
                        # a big name for the spreadsheet sheet.
                        # Sheet name can be <= 31 chars
                        if len(filter_aux) > 1:
                            code = '_' + str(c)
                        else:
                            code = ''

                        # Check if z-score data frame is empty or not
                        if signature.empty:
                            print common_functions.bcolors.WARNING + 'Warning: Dataframe is empty for p-value <= ' \
                                  + str(p_v) + ', fold change >= ' + str(f_c) + ' and <= -' + str(f_c)\
                                                                              + ', and intensity >= ' + str(m_e) \
                                  + '. No heatmap or signature sheet will be create for this data. ' \
                                  + 'Try to change the criteria in the JSON file.\n\n' \
                                  + common_functions.bcolors.ENDC
                            signature_dict[comparison_name + code] = OrderedDict()
                            signature_dict[comparison_name + code]["No. of antibodies"] = 0
                            signature_dict[comparison_name + code]["UP-regulated"] = 0
                            signature_dict[comparison_name + code]["DN-regulated"] = 0
                            signature_dict[comparison_name + code]["p-value"] = p_v
                            signature_dict[comparison_name + code]["Fold change"] = f_c
                            signature_dict[comparison_name + code]["Maximum intensity"] = m_e
                            c += 1
                            self.signature_code[comparison_name + code] = 'sig_' + str(self.sig_count)
                            self.sig_count += 1

                        else:
                            # Pass dataframe to z-score function
                            df_for_z_score = pd.concat(
                                [self.norm_data_frame.iloc[1:, :1], control_data, treatment_data],
                                axis=1, join_axes=[signature.index])

                            z_score_output = self.generate_z_scores_report(df_for_z_score, comparison_name + suffix)

                            print '==> Starting heatmap for comparison'
                            cmd = self.heatmap_script + ' -i ' + self.input + ' -z ' + z_score_output
                            print '==>Ready to execute ' + cmd + '\n'
                            result = os.system('python ' + cmd)
                            if result != 0:
                                sys.exit(cmd + ' exit ' + str(result))

                            # Write signurates to spreadsheet
                            self.write_signature(signature, control_data, treatment_data, comparison_name, code, suffix,
                                                 p_v, f_c, m_e)

                            # Creating signature combined data frame in order to add to the xlsx summary
                            if signatures_combined is None:
                                signatures_combined = signature.filter(regex=self.fold_change_prefix)
                            else:
                                signatures_combined = pd.concat([signatures_combined,
                                                                 signature.filter(regex=self.fold_change_prefix)], axis=1)
                            signature_dict[comparison_name + code] = OrderedDict()
                            signature_dict[comparison_name + code]["No. of antibodies"] = len(signature.index)
                            signature_dict[comparison_name + code]["UP-regulated"] = len((signature[signature[l3] > 0]).index)
                            signature_dict[comparison_name + code]["DN-regulated"] = len((signature[signature[l3] < 0]).index)
                            signature_dict[comparison_name + code]["p-value"] = p_v
                            signature_dict[comparison_name + code]["Fold change"] = f_c
                            signature_dict[comparison_name + code]["Maximum intensity"] = m_e
                            c += 1
                    # After creating the signature we can drop the max columns in order to print the data without
                    # no issues
                    df_aux.drop(label4, axis=1, inplace=True)

                # Calculate ANOVA
                elif self.group_comparisons[i].get("stats") and self.group_comparisons[i]["stats"] == "ANOVA":
                    norm_comp_dict = {}
                    # Adding control data to the dictionary
                    for c in range(0, len(self.group_comparisons[i]["control"])):
                        # Getting the number of elements in the matrix in order to iterate through the matrices
                        if c == 0:
                            n = len(self.grouped_data_dict[self.group_comparisons[i]["control"][c]])
                            control_data = self.grouped_data_dict[self.group_comparisons[i]["control"][c]]
                            control_names = self.group_comparisons[i]["control"][c]
                        else:
                            # If more than one control add data frame log2 to control_data
                            control_data = control_data.join(self.grouped_data_dict[self.group_comparisons[i]["control"][c]])
                            control_names += '_' + self.group_comparisons[i]["control"][c]
                        # Normalizing data
                        self.check_for_zero(self.grouped_data_dict[self.group_comparisons[i]["control"][c]])
                        norm_aux = np.log2(self.grouped_data_dict[self.group_comparisons[i]["control"][c]])
                        # Save normalized matrix in the dictionary
                        norm_comp_dict[self.group_comparisons[i]["control"][c]] = norm_aux.as_matrix()
                    # Adding treatment data to the dictionary
                    for t in range(0, len(self.group_comparisons[i]["test"])):
                        if t == 0:
                            treatment_data = self.grouped_data_dict[self.group_comparisons[i]["test"][t]]
                            treatment_names = self.group_comparisons[i]["test"][t]
                        else:
                            treatment_data = treatment_data.join(
                                self.grouped_data_dict[self.group_comparisons[i]["test"][t]])
                            treatment_names += '_' + self.group_comparisons[i]["test"][t]
                        # Normalizing data
                        self.check_for_zero(self.grouped_data_dict[self.group_comparisons[i]["test"][t]])
                        norm_aux = np.log2(self.grouped_data_dict[self.group_comparisons[i]["test"][t]])
                        # Save normalized matrix in the dictionary
                        norm_comp_dict[self.group_comparisons[i]["test"][t]] = norm_aux.as_matrix()

                    comparison_name = treatment_names + '.over.' + control_names
                    print 'Comparison: ' + comparison_name
                    # Execute ANOVA
                    p_value = self.anova(norm_comp_dict, n)
                    # Adjust p-value
                    p_value_adjusted = self.adjust_p_value(p_value, self.correction)

                    # Creating a list of parameters in order to be able to work with
                    # one or more filters
                    # Since in python setting a variable actually sets a reference to the variable.
                    # We need to simply copy the list to the new list:
                    filter_aux = self.anova_filter_list[:]
                    if self.group_comparisons[i].get("p-value"):
                        for f in self.group_comparisons[i]['p-value']:
                            filter_aux.append(f)

                    # Creating data frame with p-value and adjusted p-value
                    label1 = self.p_value_prefix + '_' + comparison_name
                    label2 = self.adj_p_value_prefix + '_' + comparison_name
                    df_aux = pd.DataFrame({label1: p_value,
                                           label2: p_value_adjusted},
                                          index=control_data.index, columns=[label1, label2])
                    if 'ANOVA' in self.comparisons_results:
                        self.comparisons_results[self.anova_prefix][comparison_name] = df_aux
                    else:
                        self.comparisons_results[self.anova_prefix] = {}
                        self.comparisons_results[self.anova_prefix][comparison_name] = df_aux

                    c = 1
                    for p_v in filter_aux:
                        # Adding suffix to label in order to differ comparison by filter
                        suffix = '_' + str(p_v)
                        l1 = label1 + suffix
                        l2 = label2 + suffix

                        new_column = [l1, l2]
                        # df_aux.copy aims to avoid making changes in the copy
                        # Since python sets a reference to the variable instead of setting the variable
                        # Doing this when I change the name of the columns bellow it doesn't affect the
                        # data stored in self.comparisons_results (df_aux)
                        df_filtered = df_aux.copy()
                        df_filtered.columns = new_column

                        # Find values with p-value <= p_v
                        df_filtered = df_filtered.loc[df_filtered[l1] <= p_v]

                        # Check if z-score data frame is empty or not
                        if df_filtered.empty:
                            print common_functions.bcolors.WARNING + 'Warning: Dataframe is empty for p-value <= ' \
                                  + str(p_v) + '. No heatmap or anova sheet will be create for this data.' \
                                  + 'Try to change the p-value in the JSON file.\n\n' \
                                  + common_functions.bcolors.ENDC
                        else:
                            # Pass dataframe to z-score function
                            df_for_z_score = pd.concat(
                                [self.norm_data_frame.iloc[1:, :1], control_data, treatment_data],
                                axis=1, join_axes=[df_filtered.index])

                            z_score_output = self.generate_z_scores_report(df_for_z_score, self.anova_prefix + '_'
                                                                           + comparison_name + suffix)
                            print '==> Starting heatmap for comparison'
                            cmd = self.heatmap_script + ' -i ' + self.input + ' -z ' + z_score_output
                            print '==>Ready to execute ' + cmd + '\n'
                            result = os.system('python ' + cmd)
                            if result != 0:
                                sys.exit(cmd + ' exit ' + str(result))

                            # write anova results to anova summary
                            self.write_anova_summary(df_filtered, control_data, treatment_data,
                                                     comparison_name, suffix, p_v)
                    if self.anova_workbook_summary is not None:
                        self.anova_workbook_summary.close()

                else:
                    sys.exit('Unknown stats type' + self.group_comparisons[i]["stats"])
            return signatures_combined, signature_dict

    # Execute comparisons: t-test or ANOVA
    def exec_comparisons_mda_tcga(self):
        signatures_combined = None
        signature_dict = OrderedDict()
        # Comparisons
        # If group_comparisons not empty
        if self.group_comparisons:
            print '==> Starting comparisons: '

            for i in range(0, len(self.group_comparisons)):

                # Execute t-test
                if not self.group_comparisons[i].get("stats") or \
                        (self.group_comparisons[i].get("stats") and self.group_comparisons[i]["stats"] == "t-test"):
                    # Normalize data
                    # Read controls. Calculate log2
                    for c in range(0, len(self.group_comparisons[i]["control"])):
                        # Since comparison can be defined in the JSON file with spaces or special char
                        # In order to avoid problems and to keep the data consistency
                        # We apply the same treatment to the column header we did on convertInput.py line 166
                        self.group_comparisons[i]["control"][c] = self.group_comparisons[i]["control"][c].replace(' ',
                                                                                                                  '_')
                        self.group_comparisons[i]["control"][c] = \
                            common_functions.replace_chars(self.group_comparisons[i]["control"][c])

                        if c == 0:
                            # The data from MDA is already normalized log2
                            control_data_log2 = self.grouped_data_dict[self.group_comparisons[i]["control"][c]]
                            control_names = self.group_comparisons[i]["control"][c]
                        else:
                            # The data from MDA is already normalized log2
                            # If more than one control add data frame log2 to control_data
                            control_data_log2 = control_data_log2.join(self.grouped_data_dict[self.group_comparisons[i]["control"][c]])
                            control_names += '_' + self.group_comparisons[i]["control"][c]
                    # Read treatments. Calculate log2

                    for t in range(0, len(self.group_comparisons[i]["test"])):
                        # Since comparison can be defined in the JSON file with spaces or special char
                        # In order to avoid problems and to keep the data consistency
                        # We apply the same treatment to the column header we did on convertInput.py line 166
                        self.group_comparisons[i]["test"][t] = self.group_comparisons[i]["test"][t].replace(' ', '_')
                        self.group_comparisons[i]["test"][t] = common_functions.replace_chars(self.group_comparisons[i]["test"][t])

                        if t == 0:
                            # The data from MDA is already normalized log2
                            treatment_data_log2 = self.grouped_data_dict[self.group_comparisons[i]["test"][t]]
                            treatment_names = self.group_comparisons[i]["test"][t]
                        else:
                            # The data from MDA is already normalized log2
                            # If more than one control add data frame log2 to control_data
                            treatment_data_log2 = treatment_data_log2.join(self.grouped_data_dict[self.group_comparisons[i]["test"][t]])
                            treatment_names += '_' + self.group_comparisons[i]["test"][t]

                    # Creating comparison name. Test come first.
                    comparison_name = treatment_names + self.COMP_INFIX + control_names

                    print '===> Comparison: ' + comparison_name
                    # Execute t-test
                    (p_value, fold_change) = self.t_test(control_data_log2, treatment_data_log2)

                    # Adjust p-value
                    p_value_adjusted = self.adjust_p_value(p_value, self.correction)

                    # Creating a list of parameters in order to be able to work with signatures
                    # for one or more filters
                    # Since in python setting a variable actually sets a reference to the variable.
                    # We need to simply copy the list to the new list:
                    filter_aux = self.t_test_filter_list[:]
                    if self.group_comparisons[i].get("filter"):
                        for f in self.group_comparisons[i]['filter']:
                            filter_aux.append(f)
                    # Creating data frame with p-value, fold change and ajusted p-value
                    label1 = self.p_value_prefix + '_' + comparison_name
                    label2 = self.adj_p_value_prefix + '_' + comparison_name
                    label3 = self.fold_change_prefix + '_' + comparison_name
                    df_aux = pd.DataFrame({label1: p_value,
                                           label2: p_value_adjusted,
                                           label3: fold_change},
                                          index=control_data_log2.index, columns=[label1, label2, label3])
                    if 't-test' in self.comparisons_results:
                        self.comparisons_results[self.t_test_prefix][comparison_name] = df_aux
                    else:
                        self.comparisons_results[self.t_test_prefix] = {}
                        self.comparisons_results[self.t_test_prefix][comparison_name] = df_aux

                    c = 1
                    for f in filter_aux:
                        if 'p-value' not in f:
                            self.p_value
                        else:
                            p_v = f['p-value']
                        if 'fold_change' not in f:
                            f_c = self.fold_change
                        else:
                            f_c = f['fold_change']

                        if 'intensity' not in f:
                            m_e = self.max_intensity
                        else:
                            m_e = f['intensity']

                        # Adding suffix to label in order to differ comparison by filter
                        suffix = '_' + str(p_v) + '-' + str(f_c)
                        l1 = label1 + suffix
                        l2 = label2 + suffix
                        l3 = label3 + suffix

                        new_column = [l1, l2, l3]
                        # df_aux.copy aims to avoid making changes in the copy
                        # Since python sets a reference to the variable instead of setting the variable
                        # Doing this when I change the name of the columns bellow it doesn't affect the
                        # data stored in self.comparisons_results (df_aux)
                        df_sig = df_aux.copy()
                        df_sig.columns = new_column

                        # Find signature for dataframe
                        signature = self.find_signatures(df_sig, p_v, f_c, None)

                        # code is used in order to identify the comparison and its parameters without creating
                        # a big name for the spreadsheet sheet.
                        # Sheet name can be <= 31 chars
                        if len(filter_aux) > 1:
                            code = '_' + str(c)
                        else:
                            code = ''

                        # Check if z-score data frame is empty or not
                        if signature.empty:
                            print common_functions.bcolors.WARNING + 'Warning: Dataframe is empty for p-value <= ' \
                                  + str(p_v) + ', fold change >= ' + str(f_c) + ' and <= -' + str(f_c)\
                                  + '. No heatmap or signature sheet will be create for this data. ' \
                                  + 'Try to change the criteria in the JSON file.\n\n' \
                                  + common_functions.bcolors.ENDC
                            signature_dict[comparison_name + code] = OrderedDict()
                            signature_dict[comparison_name + code]["No. of antibodies"] = 0
                            signature_dict[comparison_name + code]["UP-regulated"] = 0
                            signature_dict[comparison_name + code]["DN-regulated"] = 0
                            signature_dict[comparison_name + code]["p-value"] = p_v
                            signature_dict[comparison_name + code]["Fold change"] = f_c
                            signature_dict[comparison_name + code]["Maximum intensity"] = m_e
                            c += 1
                            self.signature_code[comparison_name + code] = 'sig_' + str(self.sig_count)
                            self.sig_count += 1

                        else:
                            # Pass dataframe to z-score function
                            df_for_z_score = pd.concat(
                                [self.norm_data_frame.iloc[0:, :1], control_data_log2, treatment_data_log2],
                                axis=1, join_axes=[signature.index])
                            z_score_output = self.generate_z_scores_report(df_for_z_score, comparison_name + suffix)

                            print '==> Starting heatmap for comparison'
                            cmd = self.heatmap_script + ' -i ' + self.input + ' -z ' + z_score_output
                            print '==>Ready to execute ' + cmd + '\n'
                            result = os.system('python ' + cmd)
                            if result != 0:
                                sys.exit(cmd + ' exit ' + str(result))

                            # Write signurates to spreadsheet
                            self.write_signature_mda(signature, control_data_log2, treatment_data_log2, comparison_name,
                                                     code, suffix, p_v, f_c)

                            # Creating signature combined data frame in order to add to the xlsx summary
                            if signatures_combined is None:
                                signatures_combined = signature.filter(regex=self.fold_change_prefix)
                            else:
                                signatures_combined = pd.concat([signatures_combined,
                                                                 signature.filter(regex=self.fold_change_prefix)], axis=1)

                            signature_dict[comparison_name + code] = OrderedDict()
                            signature_dict[comparison_name + code]["No. of antibodies"] = len(signature.index)
                            signature_dict[comparison_name + code]["UP-regulated"] = len((signature[signature[l3] > 0]).index)
                            signature_dict[comparison_name + code]["DN-regulated"] = len((signature[signature[l3] < 0]).index)
                            signature_dict[comparison_name + code]["p-value"] = p_v
                            signature_dict[comparison_name + code]["Fold change"] = f_c
                            signature_dict[comparison_name + code]["Maximum intensity"] = m_e
                            c += 1

                # Calculate ANOVA
                elif self.group_comparisons[i].get("stats") and self.group_comparisons[i]["stats"] == "ANOVA":
                    norm_comp_dict = {}
                    # Adding control data to the dictionary
                    for c in range(0, len(self.group_comparisons[i]["control"])):
                        # Getting the number of elements in the matrix in order to iterate through the matrices
                        if c == 0:
                            n = len(self.grouped_data_dict[self.group_comparisons[i]["control"][c]])
                            control_data_log2 = self.grouped_data_dict[self.group_comparisons[i]["control"][c]]
                            control_names = self.group_comparisons[i]["control"][c]
                        else:
                            # If more than one control add data frame log2 to control_data
                            control_data_log2 = control_data_log2.join(self.grouped_data_dict[self.group_comparisons[i]["control"][c]])
                            control_names += '_' + self.group_comparisons[i]["control"][c]
                        # Save normalized matrix in the dictionary
                        norm_comp_dict[self.group_comparisons[i]["control"][c]] = control_data_log2.as_matrix()
                    # Adding treatment data to the dictionary
                    for t in range(0, len(self.group_comparisons[i]["test"])):
                        if t == 0:
                            treatment_data_log2 = self.grouped_data_dict[self.group_comparisons[i]["test"][t]]
                            treatment_names = self.group_comparisons[i]["test"][t]
                        else:
                            treatment_data_log2 = treatment_data_log2.join(
                                self.grouped_data_dict[self.group_comparisons[i]["test"][t]])
                            treatment_names += '_' + self.group_comparisons[i]["test"][t]

                        # Save normalized matrix in the dictionary
                        norm_comp_dict[self.group_comparisons[i]["test"][t]] = treatment_data_log2.as_matrix()

                    comparison_name = treatment_names + self.COMP_INFIX + 'x' + control_names
                    print 'Comparison: ' + comparison_name
                    # Execute ANOVA
                    p_value = self.anova(norm_comp_dict, n)
                    # Adjust p-value
                    p_value_adjusted = self.adjust_p_value(p_value, self.correction)

                    # Creating a list of parameters in order to be able to work with
                    # one or more filters
                    # Since in python setting a variable actually sets a reference to the variable.
                    # We need to simply copy the list to the new list:
                    filter_aux = self.anova_filter_list[:]
                    if self.group_comparisons[i].get("p-value"):
                        for f in self.group_comparisons[i]['p-value']:
                            filter_aux.append(f)

                    # Creating data frame with p-value and adjusted p-value
                    label1 = self.p_value_prefix + '_' + comparison_name
                    label2 = self.adj_p_value_prefix + '_' + comparison_name
                    df_aux = pd.DataFrame({label1: p_value,
                                           label2: p_value_adjusted},
                                          index=control_data_log2.index, columns=[label1, label2])
                    if 'ANOVA' in self.comparisons_results:
                        self.comparisons_results[self.anova_prefix][comparison_name] = df_aux
                    else:
                        self.comparisons_results[self.anova_prefix] = {}
                        self.comparisons_results[self.anova_prefix][comparison_name] = df_aux
                    c = 1
                    for p_v in filter_aux:
                        # Adding suffix to label in order to differ comparison by filter
                        suffix = '_' + str(p_v)
                        l1 = label1 + suffix
                        l2 = label2 + suffix

                        new_column = [l1, l2]
                        # df_aux.copy aims to avoid making changes in the copy
                        # Since python sets a reference to the variable instead of setting the variable
                        # Doing this when I change the name of the columns bellow it doesn't affect the
                        # data stored in self.comparisons_results (df_aux)
                        df_filtered = df_aux.copy()
                        df_filtered.columns = new_column

                        # Find values with p-value <= p_v
                        df_filtered = df_filtered.loc[df_filtered[l1] <= p_v]

                        # Check if z-score data frame is empty or not
                        if df_filtered.empty:
                            print common_functions.bcolors.WARNING + 'Warning: Dataframe is empty for p-value <= ' \
                                  + str(p_v) + '. No heatmap or anova sheet will be create for this data.' \
                                  + 'Try to change the p-value in the JSON file.\n\n' \
                                  + common_functions.bcolors.ENDC
                        else:
                            # Pass dataframe to z-score function
                            df_for_z_score = pd.concat(
                                [self.norm_data_frame.iloc[0:, :1], control_data_log2, treatment_data_log2],
                                axis=1, join_axes=[df_filtered.index])

                            z_score_output = self.generate_z_scores_report(df_for_z_score, self.anova_prefix + '_'
                                                                           + comparison_name + suffix)
                            print '==> Starting heatmap for comparison'
                            cmd = self.heatmap_script + ' -i ' + self.input + ' -z ' + z_score_output
                            print '==>Ready to execute ' + cmd + '\n'
                            result = os.system('python ' + cmd)
                            if result != 0:
                                sys.exit(cmd + ' exit ' + str(result))

                            # write anova results to anova summary
                            self.write_anova_summary(df_filtered, control_data_log2, treatment_data_log2,
                                                     comparison_name, suffix, p_v)
                    if self.anova_workbook_summary is not None:
                        self.anova_workbook_summary.close()

                else:
                    sys.exit('Unknown stats type' + self.group_comparisons[i]["stats"])
            return signatures_combined, signature_dict

    def write_anova_summary(self, stats_df, c_df, t_df, comparison, suffix, p_v):
        print '==> Writng ANOVA summary for ' + comparison
        if self.anova_workbook_summary is None:
            output = self.output_dir + '/' + self.anova_prefix + '_summary.xlsx'
            self.anova_workbook_summary = xlsxwriter.Workbook(output)
            self.anova_format = common_functions.set_formats(self.anova_workbook_summary)
        # In order to avoid the error:
        # Exception: Excel worksheet name 'Weight_loss_+_aero_exe_BL_vs_control_BL' must be <= 31 chars.
        # The name of the spread sheet name is reduced to 29 chars + ..
        sheet_name = comparison + suffix
        sheet_name = (comparison[:25] + '..' + str(p_v)) if len(sheet_name) > 29 else sheet_name
        worksheet = self.anova_workbook_summary.add_worksheet(sheet_name)

        # Last number from 0 of descrition column for BCM data
        last_desc_column = 4
        start = 1
        if self.input_format == 'MDA':
            # Last number from 0 of descrition column for MDAnderson data
            start = 0
            last_desc_column = 6

        # Gathering the information for the indeces in the signature
        # New dataframe conatins: description, treatment and control median values, stats results
        # Length of description dataframe (self.norm_data_frame.iloc[1:, :4].columns)
        m = len(self.norm_data_frame.iloc[start:, :last_desc_column].columns)
        # Length of stats dataframe (stats_df)
        n = len(stats_df.columns)

        df = pd.concat([self.norm_data_frame.iloc[start:, :last_desc_column], t_df, c_df, stats_df], axis=1, join_axes=[stats_df.index])

        level_0 = df.columns
        level_1 = df.columns.tolist()

        if self.input_format == 'BCM':
            # Writing AB_ID in the header
            # Write indexes
            worksheet.write(1, 0, self.index_header, self.anova_format['1'])
        else:
            # Writing Slide_no in the header
            # Write indexes
            worksheet.write(1, 0, df.index.name, self.anova_format['1'])
        for i in range(0, len(df)):
            worksheet.write(i + 2, 0, df.index[i], self.anova_format['bold'])

        # Write column header
        fold_change_column = None

        for c in range(0, len(df.columns)):
            if c == m - 1:
                # Add right border header
                worksheet.write(0, c + 1, '', self.anova_format['3'])
                # Bold with bottom and right border header
                worksheet.write(1, c + 1, df.columns[c], self.anova_format['4'])
            # Print bio rep name in the first row
            elif (c >= m) & (c < (len(df.columns) - n)):
                # Add right border
                if c == (len(df.columns) - n - 1):
                    if self.header_dict is not None and self.header_dict:
                        worksheet.write(0, c + 1, df.columns[c], self.anova_format['11'])
                        worksheet.write(1, c + 1, self.header_dict[df.columns[c]], self.anova_format['10'])
                        level_1[c] = self.header_dict[df.columns[c]]
                    else:
                        worksheet.write(0, c + 1, "", self.anova_format['11'])
                        worksheet.write(1, c + 1, df.columns[c], self.anova_format['10'])
                else:
                    if self.header_dict is not None and self.header_dict:
                        worksheet.write(0, c + 1, df.columns[c], self.anova_format['7'])
                        worksheet.write(1, c + 1, self.header_dict[df.columns[c]], self.anova_format['8'])
                        level_1[c] = self.header_dict[df.columns[c]]
                    else:
                        worksheet.write(1, c + 1, df.columns[c], self.anova_format['8'])
                worksheet.set_column(c + 1, c + 1, 12)
            else:
                # Bold with bottom border header
                # Remove comparison name from the header
                header = df.columns[c]
                header = header.replace('_' + comparison, '')
                # Add new format
                f = self.anova_workbook_summary.add_format({'text_wrap': True})
                f.set_align('center')
                f.set_align('vcenter')
                f.set_bottom()
                f.set_bold()
                if self.adj_p_value_prefix in header:
                    worksheet.write(1, c + 1, self.adj_p_value_prefix, f)
                    # Increase column width
                    worksheet.set_column(c + 1, c + 1, 18)
                elif self.p_value_prefix in header:
                    worksheet.write(1, c + 1, self.p_value_prefix + '\n (<= ' + str(p_v) + ')', f)
                    # Increase column width
                    worksheet.set_column(c + 1, c + 1, 12)
                    # Increase row height
                    worksheet.set_row(1, 38)
                else:
                    worksheet.write(1, c + 1, header, self.anova_format['8'])
                    # Increase column width
                    worksheet.set_column(c + 1, c + 1, 14)
                if c >= (len(df.columns) - n):
                    level_1[c] = self.nan
        # Write stats comparison name in a merged cell
        f = self.anova_workbook_summary.add_format({'text_wrap': True})
        # Foarmatting merged cell
        f.set_align('center')
        f.set_align('vcenter')
        f.set_bold()
        # Increase row height
        worksheet.set_row(0, 38)
        worksheet.merge_range(0, len(df.columns) - n + 1, 0, len(df.columns), comparison, f)

        # Write signature data
        for r in range(0, len(df)):
            for c in range(0, len(df.columns)):
                if (c == m - 1) or (c == (len(df.columns) - n - 1)):
                    # Print bio description
                    if c >= m:
                        worksheet.write(r + 2, c + 1, df.iloc[r, c], self.anova_format['6'])
                    # Print
                    else:
                        worksheet.write(r + 2, c + 1, df.iloc[r, c], self.anova_format['5'])
                else:
                    # Print bio description
                    if c >= m:
                        worksheet.write(r + 2, c + 1, df.iloc[r, c], self.anova_format['9'])
                    else:
                        worksheet.write(r + 2, c + 1, df.iloc[r, c])

        # Adding image to sheet
        pattern = '.*' + self.anova_prefix + '_' + comparison + suffix + '.*' + self.color_map_distance
        heatmap_file = self.find_heatmap_files(pattern)
        if heatmap_file is not None and os.path.isfile(heatmap_file):
            worksheet.insert_image(3, c + 3, heatmap_file,
                                   {'x_scale': self.scale_heatmap, 'y_scale': self.scale_heatmap})

        # Write data to xlsx
        # Since writing multilevel dataframes to csv adds a blank line after the multilevel
        # I had to make a workaround this issue.
        # So I'm creating a dataframe with level_1 as the data and level_0 as the columns names
        # Then, I concat this dataframe with the df created previously

        print '==> Writing bio rep data to xlsx'
        if self.input_format == 'BCM':
            aux_df = pd.DataFrame(data=[level_1], columns=level_0, index=[self.index_header])
            frames = [aux_df, df]
            aux_df = pd.concat(frames)
            # To keep the order of the indexes
            aux_index = [self.index_header] + df.index.tolist()
            aux_df.reindex(aux_index)

            aux_df.to_csv(self.output_dir + '/' + self.anova_prefix + '_' + comparison + suffix + '.csv', sep="\t",
                          na_rep=self.nan, index=True)
        else:
            df.to_csv(self.output_dir + '/' + self.anova_prefix + '_' + comparison + suffix + '.csv', sep="\t",
                  na_rep=self.nan, index=True)

    def find_signatures(self, df, p_v, f_c, m_e):
        print '==> Finding signatures'
        if f_c == 0:
            sys.exit(common_functions.bcolors.FAIL + "Error: fold change can't be zero.")

        # Finding signature for BCM data. Use intensity
        if m_e is not None:
            label1 = df.filter(regex=self.p_value_prefix).columns[0]
            label2 = df.filter(regex=self.fold_change_prefix).columns[0]
            label3 = df.filter(regex=self.max_intensity_prefix).columns[0]
            signature = df.loc[(abs(df[label2]) >= np.log2(f_c)) & (df[label1] <= p_v) & (df[label3] > m_e)]
            # Remove max column from dataframe
            # At first I was using signature.drop(label3, axis=1, inplace=True)
            # The inplace=True deletes the column without having to reassign
            # However I was getting the SettingWithCopyWarning earning
            # I had to reassing instead of using inplace
            signature = signature.drop(label3, axis=1)
            # Find signature for MDAnderson data
        else:
            label1 = df.filter(regex=self.p_value_prefix).columns[0]
            label2 = df.filter(regex=self.fold_change_prefix).columns[0]
            signature = df.loc[(abs(df[label2]) >= np.log2(f_c)) & (df[label1] <= p_v)]
        return signature

    def find_heatmap_files(self, pattern):
        print '==> Finding heatmap files for pattern: ' + pattern
        h_file = None
        heatmap_pattern = pattern + '.*' + self.heatmap_extension
        for f in os.listdir(self.output_dir):
            if re.match(heatmap_pattern, f):
                h_file = self.output_dir + '/' + f
        return h_file

    def write_signature(self, stats_df, c_df, t_df, comparison, code, suffix, p_v, f_c, m_e):
        print '==> Writing signature to summary for signature ' + comparison + suffix
        # In order to avoid the error:
        # Exception: Excel worksheet name 'Weight_loss_+_aero_exe_BL_vs_control_BL' must be <= 31 chars.
        # The name of the spread sheet name is reduced to 29 chars + ..
        sheet_name = self.signature_prefix + comparison + code
        #sheet_name = ((self.signature_prefix + comparison)[:28] + '..' + str(code)) if len(sheet_name) > 29 else sheet_name

        self.signature_code[comparison + code] = 'sig_' + str(self.sig_count)
        self.sig_count += 1

        # worksheet = self.workbook_summary.add_worksheet(sheet_name)
        #
        # self.sig_sheet_list.append(sheet_name)

        worksheet = self.workbook_summary.add_worksheet(self.signature_code[comparison + code])

        self.sig_sheet_list.append(self.signature_code[comparison + code])

        # Gathering the information for the indeces in the signature
        # New dataframe conatins: description, treatment and control median values, stats results
        # Length of description dataframe (self.norm_data_frame.iloc[1:, :4].columns)
        m = len(self.norm_data_frame.iloc[1:, :4].columns)
        # Length of stats dataframe (stats_df)
        n = len(stats_df.columns)
        df = pd.concat([self.norm_data_frame.iloc[1:, :4], t_df, c_df, stats_df], axis=1, join_axes=[stats_df.index])

        # find log2 column name to used it to sort the data by log2 fold change
        filtered = filter(lambda x: self.fold_change_prefix in x, df.columns)
        df.sort_values(filtered, inplace=True, ascending=False)

        level_0 = df.columns
        level_1 = df.columns.tolist()

        # Writing AB_ID in the header
        # Write indexes
        worksheet.write(1, 0, self.index_header, self.format['1'])
        for i in range(0, len(df)):
            worksheet.write(i + 2, 0, df.index[i], self.format['bold'])

        # Write column header
        fold_change_column = None
        for c in range(0, len(df.columns)):
            if c == m - 1:
                # Add right border header
                worksheet.write(0, c + 1, '', self.format['3'])
                # Bold with bottom and right border header
                worksheet.write(1, c + 1, df.columns[c], self.format['4'])
            # Print bio rep name in the first row
            elif (c >= m) & (c < (len(df.columns) - n)):
                # Add right border
                if c == (len(df.columns) - n - 1):
                    worksheet.write(0, c + 1, df.columns[c], self.format['11'])
                    worksheet.write(1, c + 1, self.header_dict[df.columns[c]], self.format['10'])
                    level_1[c] = self.header_dict[df.columns[c]]
                else:
                    worksheet.write(0, c + 1, df.columns[c], self.format['7'])
                    worksheet.write(1, c + 1, self.header_dict[df.columns[c]], self.format['8'])
                    level_1[c] = self.header_dict[df.columns[c]]
                worksheet.set_column(c + 1, c + 1, 12)
            else:
                # Bold with bottom border header
                # Remove comparison name from the header
                header = df.columns[c]
                header = header.replace('_' + comparison, '')
                # Add new format
                f = self.workbook_summary.add_format({'text_wrap': True})
                f.set_align('center')
                f.set_align('vcenter')
                f.set_bottom()
                f.set_bold()
                if self.adj_p_value_prefix in header:
                    worksheet.write(1, c + 1, self.adj_p_value_prefix, f)
                    # Increase column width
                    worksheet.set_column(c + 1, c + 1, 18)
                elif self.p_value_prefix in header:
                    worksheet.write(1, c + 1, self.p_value_prefix + '\n (<= ' + str(p_v) + ')', f)
                    # Increase column width
                    worksheet.set_column(c + 1, c + 1, 12)
                    # Increase row height
                    worksheet.set_row(1, 38)
                elif self.fold_change_prefix in header:
                    worksheet.write(1, c + 1, self.fold_change_prefix + '\n log2(' + str(f_c) + '), <= -log2('
                                    + str(f_c) + '))', f)
                    fold_change_column = c + 1
                    # Increase column width
                    worksheet.set_column(c + 1, c + 1, 24)
                else:
                    worksheet.write(1, c + 1, header, self.format['8'])
                    # Increase column width
                    worksheet.set_column(c + 1, c + 1, 14)
                if c >= (len(df.columns) - n):
                    level_1[c] = self.nan
        # Write stats comparison name in a merged cell
        f = self.workbook_summary.add_format({'text_wrap': True})
        # Foarmatting merged cell
        f.set_align('center')
        f.set_align('vcenter')
        f.set_bold()
        # Increase row height
        worksheet.set_row(0, 38)
        worksheet.merge_range(0, len(df.columns) - n + 1, 0, len(df.columns),
                              comparison + '\n max. int. > ' + str(m_e), f)

        # Write signature data
        for r in range(0, len(df)):
            for c in range(0, len(df.columns)):
                # Paint background and font according to the fold change intensity
                if c == fold_change_column - 1:
                    f = self.workbook_summary.add_format()
                    bg_color, font_color = self.get_background_and_font_color(df.iloc[r, c])
                    f.set_bg_color(bg_color)
                    f.set_align('center')
                    f.set_font_color(font_color)
                    worksheet.write(r + 2, c + 1, df.iloc[r, c], f)
                elif (c == m - 1) or (c == (len(df.columns) - n - 1)):
                    # Print bio description
                    if c >= m:
                        worksheet.write(r + 2, c + 1, df.iloc[r, c], self.format['6'])
                    # Print
                    else:
                        worksheet.write(r + 2, c + 1, df.iloc[r, c], self.format['5'])
                else:
                    # Print bio description
                    if c >= m:
                        worksheet.write(r + 2, c + 1, df.iloc[r, c], self.format['9'])
                    else:
                        worksheet.write(r + 2, c + 1, df.iloc[r, c])

        # Adding image to sheet
        pattern = '.*' + comparison + suffix + '.*' + self.color_map_distance
        heatmap_file = self.find_heatmap_files(pattern)
        if heatmap_file is not None and os.path.isfile(heatmap_file):
            worksheet.insert_image(3, c + 3, heatmap_file,
                                   {'x_scale': self.scale_heatmap, 'y_scale': self.scale_heatmap})

        # Write data to xlsx
        # Since writing multilevel dataframes to csv adds a blank line after the multilevel
        # I had to make a workaround this issue.
        # So I'm creating a dataframe with level_1 as the data and level_0 as the columns names
        # Then, I concat this dataframe with the df created previously
        print '==> Writing signature data to xlsx'
        aux_df = pd.DataFrame(data=[level_1], columns=level_0, index=[self.index_header])
        frames = [aux_df, df]
        aux_df = pd.concat(frames)
        # To keep the order of the indexes
        aux_index = [self.index_header] + df.index.tolist()
        aux_df.reindex(aux_index)

        aux_df.to_csv(self.output_dir + '/' + self.signature_prefix + '_' + comparison + suffix + '.csv', sep="\t",
                      na_rep=self.nan, index=True)

    def write_signature_mda(self, stats_df, c_df, t_df, comparison, code, suffix, p_v, f_c):
        print '==> Writing signature to summary for signature (MDA) ' + comparison + suffix
        # In order to avoid the error:
        # Exception: Excel worksheet name 'Weight_loss_+_aero_exe_BL_vs_control_BL' must be <= 31 chars.
        # The name of the spread sheet name is reduced to 29 chars + ..
        #sheet_name = self.signature_prefix + comparison + code
        #sheet_name = ((self.signature_prefix + comparison)[:28] + '..' + str(code)) if len(sheet_name) > 29 else sheet_name

        self.signature_code[comparison + code] = 'sig_' + str(self.sig_count)
        self.sig_count += 1

        worksheet = self.workbook_summary.add_worksheet(self.signature_code[comparison + code])

        self.sig_sheet_list.append(self.signature_code[comparison + code])

        # Gathering the information for the indeces in the signature
        # New dataframe conatins: description, treatment and control median values, stats results
        # Length of description dataframe (self.norm_data_frame.iloc[1:, :4].columns)
        m = len(self.norm_data_frame.iloc[1:, :6].columns)
        # Length of stats dataframe (stats_df)
        n = len(stats_df.columns)

        df = pd.concat([self.norm_data_frame.iloc[0:, :6], t_df, c_df, stats_df], axis=1, join_axes=[stats_df.index])

        # find log2 column name to used it to sort the data by log2 fold change
        filtered = filter(lambda x: self.fold_change_prefix in x, df.columns)
        df.sort_values(filtered, inplace=True, ascending=False)

        level_0 = df.columns

        # Writing AB_ID in the header
        # Write indexes
        worksheet.write(0, 0, df.index.name, self.format['1'])
        for i in range(0, len(df)):
            worksheet.write(i + 1, 0, df.index[i], self.format['bold'])

        # Write column header
        fold_change_column = None
        for c in range(0, len(df.columns)):
            if c == m - 1:
                # Bold with bottom and right border header
                worksheet.write(0, c + 1, df.columns[c], self.format['4'])
            # Print bio rep name in the first row
            elif (c >= m) & (c < (len(df.columns) - n)):
                # Add right border
                if c == (len(df.columns) - n - 1):
                    worksheet.write(0, c + 1, df.columns[c], self.format['10'])
                else:
                    worksheet.write(0, c + 1, df.columns[c], self.format['8'])
                worksheet.set_column(c + 1, c + 1, 12)
            else:
                # Bold with bottom border header
                # Remove comparison name from the header
                header = df.columns[c]
                header = header.replace('_' + comparison, '')
                # Add new format
                f = self.workbook_summary.add_format({'text_wrap': True})
                f.set_align('center')
                f.set_align('vcenter')
                f.set_bottom()
                f.set_bold()
                if self.adj_p_value_prefix in header:
                    worksheet.write(0, c + 1, self.adj_p_value_prefix, f)
                    # Increase column width
                    worksheet.set_column(c + 1, c + 1, 18)
                elif self.p_value_prefix in header:
                    worksheet.write(0, c + 1, self.p_value_prefix + '\n (<= ' + str(p_v) + ')', f)
                    # Increase column width
                    worksheet.set_column(c + 1, c + 1, 12)
                    # Increase row height
                    worksheet.set_row(0, 38)
                elif self.fold_change_prefix in header:
                    worksheet.write(0, c + 1, self.fold_change_prefix + '\n log2(' + str(f_c) + '), <= -log2('
                                    + str(f_c) + '))', f)
                    fold_change_column = c + 1
                    # Increase column width
                    worksheet.set_column(c + 1, c + 1, 24)
                else:
                    worksheet.write(0, c + 1, header, self.format['8'])
                    # Increase column width
                    worksheet.set_column(c + 1, c + 1, 14)

        # Write stats comparison name in a merged cell
        f = self.workbook_summary.add_format({'text_wrap': True})
        # Foarmatting merged cell
        f.set_align('center')
        f.set_align('vcenter')
        f.set_bold()
        # Increase row height
        worksheet.set_row(0, 38)

        # Write signature data
        for r in range(0, len(df)):
            for c in range(0, len(df.columns)):
                # Paint background and font according to the fold change intensity
                if c == fold_change_column - 1:
                    f = self.workbook_summary.add_format()
                    bg_color, font_color = self.get_background_and_font_color(df.iloc[r, c])
                    f.set_bg_color(bg_color)
                    f.set_align('center')
                    f.set_font_color(font_color)
                    worksheet.write(r + 1, c + 1, df.iloc[r, c], f)
                elif (c == m - 1) or (c == (len(df.columns) - n - 1)):
                    # Print bio description
                    if c >= m:
                        worksheet.write(r + 1, c + 1, df.iloc[r, c], self.format['6'])
                    # Print
                    else:
                        worksheet.write(r + 1, c + 1, df.iloc[r, c], self.format['5'])
                else:
                    # Print bio description
                    if c >= m:
                        worksheet.write(r + 1, c + 1, df.iloc[r, c], self.format['9'])
                    else:
                        worksheet.write(r + 1, c + 1, df.iloc[r, c])

        # Adding image to sheet
        pattern = '.*' + comparison + suffix + '.*' + self.color_map_distance
        heatmap_file = self.find_heatmap_files(pattern)
        if heatmap_file is not None and os.path.isfile(heatmap_file):
            worksheet.insert_image(3, c + 3, heatmap_file,
                                   {'x_scale': self.scale_heatmap, 'y_scale': self.scale_heatmap})

        df.to_csv(self.output_dir + '/' + self.signature_prefix + '_' + comparison + suffix + '.csv', sep="\t",
                      na_rep=self.nan, index=True)

    def generate_z_scores_report(self, df, file_name):
        print '==> Generating z-score report'
        # Since the header in df is related to the group (level 0) and not the sample (level 1)
        # We need to get the level 1 based on the level 0
        aux_header_group = [""]
        aux_header = []

        show_group = True

        # If self.header_dict exist, i.e, data is from BCM
        if self.header_dict is not None and self.header_dict and not self.technical_replication:
            for c in df.columns:
                if c in self.header_dict:
                    aux_header_group.append(self.sample_group[c])
                    aux_header.append(self.header_dict[c])
                else:
                    aux_header.append(c)
        elif self.technical_replication and show_group:
            aux_header = df.columns
            aux_dict = {}
            for c in df.columns:
                if c in self.sample_group:
                    if not self.sample_group[c] in aux_dict:
                        aux_dict[self.sample_group[c]] = []
                        aux_dict
                    aux_header_group.append(self.sample_group[c])

        # else data from MDA or TCGA
        else:
            aux_header = df.columns
            for c in df.columns:
                if c in self.sample_group:
                    aux_header_group.append(self.sample_group[c])

        # Convert dataframe in list in order to use stats.zcores
        l = df.iloc[:, 1:].values.tolist()
        z_scores = scipy.stats.zscore(l, axis=1)

        # Output file name
        z_score_output_txt = os.path.abspath(self.output_dir + '/' + self.z_score_prefix + '_' + file_name + '.txt')

        # Create txt output file
        with open(z_score_output_txt, "wb") as f:
            writer = csv.writer(f)
            # Write the group row
            writer.writerow(aux_header_group)
            writer.writerow(aux_header)
            for c in range(0, z_scores.shape[0]):
                # Add the metabolite to the row
                writer.writerow([df.iloc[c, 0]] + z_scores[c].tolist())
        return z_score_output_txt

    def write_signature_combined(self, df):
        if df is not None:
            # Last number from 0 of descrition column for BCM data
            last_desc_column = 4
            if self.input_format == 'MDA':
                # Last number from 0 of descrition column for MDAnderson data
                last_desc_column = 6

            # Adding description (AB_ID, AB_name, Slide, Gene_ID, Swiss_ID) to log2fold change info
            df1 = self.norm_data_frame.iloc[0:, :last_desc_column]
            m = len(df1.columns)
            df = pd.concat([df1, df], axis=1, join_axes=[df.index])
            # find log2 column name to used it to sort the data by log2 fold change
            filtered = filter(lambda x: self.fold_change_prefix in x, df.columns)
            df.sort_values(filtered, inplace=True, ascending=False)

            print '==> Writing combined signatures to summary'
            worksheet = self.workbook_summary.add_worksheet(self.signature_combined_prefix)
            self.new_sheet_list.append(self.signature_combined_prefix)

            # Write indexes
            worksheet.write(0, 0, '', self.format['1'])
            for i in range(0, len(df.index)):
                    worksheet.write(i + 1, 0, df.index[i], self.format['bold'])
            # Write column header
            for c in range(0, len(df.columns)):
                # Write description header
                if c < m - 1:
                    worksheet.write(0, c + 1, df.columns[c], self.format['1'])
                # Add right border to description header
                elif c == m - 1:
                    worksheet.write(0, c + 1, df.columns[c], self.format['4'])
                # Write log2_FC header
                else:
                    worksheet.write(0, c + 1, df.columns[c], self.format['2.1'])
                    # Increase height of log2_FC header
                    if c == 0:
                        worksheet.set_row(0, len(df.columns[c]) * 8)
                    elif df.columns[c] > df.columns[c - 1]:
                        worksheet.set_row(0, len(df.columns[c]) * 8)

            # Write data
            for r in range(0, len(df.index)):
                for c in range(0, len(df.columns)):
                    if c < m - 1:
                        worksheet.write(r + 1, c + 1, df.iloc[r, c])
                    # Add right border
                    elif c == m - 1:
                        worksheet.write(r + 1, c + 1, df.iloc[r, c], self.format['6'])
                    # Write fold change values
                    else:
                        if pd.isnull(df.iloc[r, c]):
                            worksheet.write_blank(r + 1, c + 1, None, self.format['6'])
                        # Add color to fold change values
                        else:
                            f = self.workbook_summary.add_format()
                            bg_color, font_color = self.get_background_and_font_color(df.iloc[r, c])
                            f.set_bg_color(bg_color)
                            f.set_align('center')
                            f.set_font_color(font_color)
                            f.set_right()
                            worksheet.write(r + 1, c + 1, df.iloc[r, c], f)

            # Write data to xlsx
            print '==> Writing combined signatures data to xlsx'
            df.to_csv(self.output_dir + '/' + self.signature_combined_prefix + '.csv', sep="\t", na_rep=self.nan, index=True)

    def write_summary_sheet(self, dct):
        # convert dct to dataframe in order to sort based on fold change
        df = pd.DataFrame.from_dict(dct)
        df = df.T
        cols = ['No. of antibodies', 'UP-regulated', 'DN-regulated', 'p-value',	'Fold change', 'Maximum intensity']

        df = df[cols]
        #df.sort_values(['Fold change'], inplace=True, ascending=False)

        # No need to write this info to a raw file. Just to the spreadsheet
        print '==> Writing summary to summary'
        worksheet = self.workbook_summary.add_worksheet(self.summary_prefix)
        worksheet.write(1, 0, 'Signature', self.format['4'])
        worksheet.set_column(0, 0, 15)
        r = 0

        #for comparison in dict:
        for r in range(0, len(df.index.values)):
            comparison = df.index.values[r]
            worksheet.write(r + 2, 0, self.signature_code[comparison], self.format['5'])
            c = 0
            for c in range(0, len(df.columns)):
                field = df.columns[c]
            #for field in dict[comparison]:

                if r == 0:
                    # Create first level header Filter criteria
                    if self.input_format == 'MDA':
                        worksheet.merge_range(0, 4, 0, 5, 'Filter criteria', self.format['7'])
                    else:
                        worksheet.merge_range(0, 4, 0, 6, 'Filter criteria', self.format['7'])
                    # Write headers
                    if c + 1 == 3:
                        worksheet.write(0, c + 1, "", self.format['5'])
                        worksheet.write(1, c + 1, field, self.format['10'])
                        # Write data
                        worksheet.write(r + 2, c + 1, dct[comparison][field], self.format['6'])
                    else:
                        worksheet.write(1, c + 1, field, self.format['8'])
                        # Write data
                        worksheet.write(r + 2, c + 1, dct[comparison][field], self.format['9'])
                else:
                    if c + 1 == 3:
                        worksheet.write(r + 2, c + 1, dct[comparison][field], self.format['6'])
                    else:
                        worksheet.write(r + 2, c + 1, dct[comparison][field], self.format['9'])
                # # If a comparison has a big name increase the cell width
                # if (c == 0) & (len(comparison) > 20):
                #     worksheet.set_column(0, 0, len(comparison) + 5)
                # Set column width according to the number of chars in the header
                worksheet.set_column(c + 1, c + 1, len(field))
                c += 1
            r += 1


    def t_test(self, c, t):
        print '===> Executing t-test'
        p_value = []
        fold_change = []

        if t.shape[1] < 2 or c.shape[1] < 2:
            sys.exit(common_functions.bcolors.FAIL + " Can't perform t-test for data with less than two groups."
                     + ' test data has ' + str(t.shape[1]) + ' group(s) while control data has ' + str(c.shape[1]))

        for i in c.index.values:
            try:
                # Calculate t-test
                p_value.append(np.around(scipy.stats.ttest_ind(c.ix[i].values, t.ix[i].values)[1], 10))
            # if not possible to perform t-test put NA in the field
            except:
                p_value.append(self.nan)
            # Calculating fold change
            fold_change.append(np.around((np.nanmean(t.ix[i]) - np.nanmean(c.ix[i])), 2))
        return p_value, fold_change

    @staticmethod
    def anova(dict, n):
        print '===> Executing ANOVA'
        p_v = []
        for r in range(0, n):
            array = []
            for comparison in dict:
                array.append(dict[comparison][r, :])
            p_v.append(scipy.stats.f_oneway(*array)[1])
        return p_v

    # Create t-test and ANOVA sheet
    # Create t-test and ANOVA xlsx
    # Create rank files for t-test
    def write_comparisons(self):
        # Write to summary
        print '==> Writing comparisons to summary'
        # Last number from 0 of description column for BCM data
        last_desc_column = 4
        if self.input_format == 'MDA':
            # Last number from 0 of description column for MDAnderson data
            last_desc_column = 6
        elif self.input_format == 'TCGA':
            # Last number from 0 of description column for TCGA data
            last_desc_column = 1

        # Create the antibodies name list
        ab_name_list = self.norm_data_frame.iloc[:, 0]

        for key in self.comparisons_results:
            print '===> ' + key
            # Creating df to write xlsx
            df = self.norm_data_frame.iloc[1:, :last_desc_column]
            m = len(df.columns)

            worksheet = self.workbook_summary.add_worksheet(key)
            self.new_sheet_list.append(key)

            if self.input_format == 'MDA' or self.input_format == 'TCGA':
                # Write index column name
                worksheet.write(1, 0, self.norm_data_frame.index.name, self.format['1'])
                # Write indexes
                for i in range(0, len(self.norm_data_frame.index)):
                        worksheet.write(i + 2, 0, self.norm_data_frame.index[i], self.format['bold'])
                # Write column header 1 until Swiss_ID
                for c in range(0, m):
                    if c == m - 1:
                        # Add right border header
                        worksheet.write(0, c + 1, '', self.format['3'])
                        # Bold with bottom and right border header
                        worksheet.write(1, c + 1, self.norm_data_frame.columns[c], self.format['4'])
                    else:
                        # Bold with bottom border header
                        worksheet.write(1, c + 1, self.norm_data_frame.columns[c], self.format['1'])
                # Write AB_name/Slide_file/Gene_ID data
                ab = None
                for c in range(0, m):
                    for r in range(0, len(self.norm_data_frame.index)):
                        if c == m - 1:
                            worksheet.write(r + 2, c + 1, self.norm_data_frame.iloc[r, c], self.format['5'])
                        else:
                            worksheet.write(r + 2, c + 1, self.norm_data_frame.iloc[r, c])
                        # Creating gene to antibody dictionary
                        if c == 0:
                            ab = self.norm_data_frame.iloc[r, c]
                            gene = self.norm_data_frame.iloc[r, c + 1]
                            gene_list = str(gene).split(',')
                            for g in gene_list:
                                if ab not in self.gene_to_ab_dict[g]:
                                    self.gene_to_ab_dict[g].append(ab)
            else:
                # Write indexes
                for i in range(0, len(self.norm_data_frame.index)):
                    if i == 0:
                        worksheet.write(i + 1, 0, self.norm_data_frame.index[i], self.format['1'])
                    else:
                        worksheet.write(i + 1, 0, self.norm_data_frame.index[i], self.format['bold'])
                # Write column header 1 until Swiss_ID
                for c in range(0, m):
                    if c == m - 1:
                        # Add right border header
                        worksheet.write(0, c + 1, '', self.format['3'])
                        # Bold with bottom and right border header
                        worksheet.write(1, c + 1, self.norm_data_frame.iloc[0, c], self.format['4'])
                    else:
                        # Bold with bottom border header
                        worksheet.write(1, c + 1, self.norm_data_frame.iloc[0, c], self.format['1'])
                # Write AB_name/Slide_file/Gene_ID data
                ab = None
                for c in range(0, m):
                    for r in range(1, len(self.norm_data_frame.index)):
                        if c == m - 1:
                            worksheet.write(r + 1, c + 1, self.norm_data_frame.iloc[r, c], self.format['5'])
                        else:
                            worksheet.write(r + 1, c + 1, self.norm_data_frame.iloc[r, c])
                        # Creating gene to antibody dictionary
                        if c == 0:
                            ab = self.norm_data_frame.iloc[r, c]
                            gene = self.norm_data_frame.iloc[r, c + 2]
                            gene_list = gene.split(',')
                            for g in gene_list:
                                if ab not in self.gene_to_ab_dict[g]:
                                    self.gene_to_ab_dict[g].append(ab)
            # Number of comparisons
            i = 0
            for comparison in self.comparisons_results[key]:
                print '===> ' + comparison
                # Contacting dataframes to print xlsx
                frames = [df, self.comparisons_results[key][comparison]]
                df = pd.concat(frames, axis=1)
                # Create rank file
                # Write header to file
                if key == self.t_test_prefix:
                    rank_file_name = self.output_dir + '/' + comparison + '.rnk'
                    rank_file = open(rank_file_name, 'w')
                    if self.input_format == 'MDA' or self.input_format == 'TCGA':
                        rank_file.write('AB_name' + '\t' + self.fold_change_prefix + '\n')
                    else:
                        rank_file.write(ab_name_list.iloc[0] + '\t' + self.fold_change_prefix + '\n')

                # Writing data to sheet
                for r in range(0, len(self.comparisons_results[key][comparison].index)):
                    n = len(self.comparisons_results[key][comparison].columns)
                    # Write AB_name and fold change to rank file
                    if key == self.t_test_prefix:
                        if self.input_format == 'MDA' or self.input_format == 'TCGA':
                            rank_file.write(ab_name_list.iloc[r] + '\t'
                                            + str(self.comparisons_results[key][comparison].iloc[r, 2]) + '\n')
                        else:
                            rank_file.write(ab_name_list.iloc[r + 1] + '\t'
                                            + str(self.comparisons_results[key][comparison].iloc[r, 2]) + '\n')
                    for c in range(0, n):
                        # Write header for each comparison
                        # (fold change, p-value, adjusted p-value) for t-test
                        # (p-value, adjusted p-value) for ANOVA
                        if r == 0:
                            header = self.comparisons_results[key][comparison].columns[c]
                            header = header.replace('_' + comparison, '')
                            if c == n - 1:
                                f.set_right()
                                worksheet.write(1, m + 1 + (n * i) + c, header, self.format['10'])
                            else:
                                worksheet.write(1, m + 1 + (n * i) + c, header, self.format['8'])
                            # Increasing the columns width
                            if key == self.anova_prefix:
                                worksheet.set_column(m + 1 + (n * i) + c, m + 1 + (n * i) + c, (len(comparison)/2) + 2)
                            else:
                                worksheet.set_column(m + 1 + (n * i) + c, m + 1 + (n * i) + c, len(header) + 2)
                        # Write name of the comparison in a merged cell
                        if r == 0 and c == 0:
                            f = self.workbook_summary.add_format()
                            f.set_bold()
                            f.set_right()
                            f.set_align('center')
                            if i % 2 == 0:
                                f.set_bg_color(self.header_background_1)
                                worksheet.merge_range(0, m + 1 + (n * i), 0, m + 1 + (n * i) + n - 1, comparison, f)
                            else:
                                f.set_bg_color(self.header_background_2)
                                worksheet.merge_range(0, m + 1 + (n * i), 0, m + 1 + (n * i) + n - 1, comparison, f)
                        # Add right border
                        if c == n - 1:
                            if pd.isnull(self.comparisons_results[key][comparison].iloc[r, c]):
                                worksheet.write_blank(r + 2, m + 1 + (n * i) + c, None, self.format['6'])
                            else:
                                worksheet.write(r + 2, m + 1 + (n * i) + c,
                                                self.comparisons_results[key][comparison].iloc[r, c], self.format['6'])
                        # No border
                        else:
                            if pd.isnull(self.comparisons_results[key][comparison].iloc[r, c]):
                                worksheet.write_blank(r + 2, m + 1 + (n * i) + c, None, self.format['9'])
                            else:
                                worksheet.write(r + 2, m + 1 + (n * i) + c,
                                                self.comparisons_results[key][comparison].iloc[r, c], self.format['9'])
                i += 1
                # Close rank file
                if key == self.t_test_prefix:
                    rank_file.close()
            # Write data to xlsx
            print '==> Writing bio rep data to xlsx'
            df.to_csv(self.output_dir + '/' + key + '.csv', sep="\t", na_rep=self.nan, index=True)

    @staticmethod
    # # c("holm", "hochberg", "hommel", "bonferroni", "BH", "BY",
#   "fdr", "none")
    def adjust_p_value(p_v, correction):
        adj_p_value = stats1.p_adjust(FloatVector(p_v), method=correction)
        return adj_p_value

    def create_PMT_file(self):
        for f in self.pathway_files:
            gmt_file = f['file']
            source = f['source']

            # Since the gmt_file has different numbers of columns, we can not use pd.read_table to read it
            # This function will think that the first number of columns is the the number of columns for all
            # rows. When a row has more columns than the first row, we'll get the following error:
            # pandas.errors.ParserError: Error tokenizing data. C error: Expected 64 fields in line 12, saw 137
            # In this error the first row started with 64 coluns but line 12 has 137 columns
            # In order to avoid such error, we read the file to a list and then use the list to build a dataframe
            # pathway_df = pd.read_table(gmt_file, header=None)
            gmt_list = [line.rstrip().split('\t') for line in open(gmt_file, 'r')]

            pathway_df = pd.DataFrame(gmt_list)

            file_name = self.output_dir + '/' + self.project_name + '_' + source + '.pmt'
            pmt_file = open(file_name, 'w')
            for r in range(0, pathway_df.shape[0]):
                start = 1
                for c in range(2, pathway_df.shape[1]):
                    gene = pathway_df.iloc[r, c]
                    if pd.notnull(gene):
                        if gene in self.gene_to_ab_dict:
                            if start == 1:
                                pmt_file.write(pathway_df.iloc[r, 0] + '\t')
                                pmt_file.write(pathway_df.iloc[r, 1])
                                start = 0
                            for protein in self.gene_to_ab_dict[gene]:
                                pmt_file.write('\t' + protein)
                if start == 0:
                    pmt_file.write('\n')
            pmt_file.close()
            #








########################################################################################
# MAIN
########################################################################################
if __name__ == '__main__':
    try:
        args = rppaStep1.get_parameters()
        if args is None:
            pass

        else:
            job = rppaStep1(args)
            job.set_input()
            job.run_step1()
            job.create_PMT_file()

    except:
        print "An unknown error occurred.\n"
        raise