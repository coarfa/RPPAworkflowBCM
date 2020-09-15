import os
import sys
import argparse
import re
import numpy as np
import scipy
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as dist
import pandas as pd
import matplotlib as mpl
from RPPA_class import RPPA
mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats
from collections import OrderedDict
import fnmatch
import shutil
import xlsxwriter
from matplotlib.backends.backend_pdf import PdfPages
import Image

def parse_args():
    parser = argparse.ArgumentParser(
        description='Calculation of signature score for samples')
    parser.add_argument('-sl', '--slide_table', type=str, required=True,
                        help='Slide table')
    parser.add_argument('-a', '--ant_dbase', type=str, required=True,
                        help='Antibody database')
    parser.add_argument('-p1', '--path', type=str, required=True,
                        help='path')
    parser.add_argument('-qp1', '--qpath', type=str, required=False,
                        help='Path for Quincy files')
    parser.add_argument('-p2', '--pmt', nargs='+', type=str, required=True,
                        help='PMT settings')
    parser.add_argument('-st', '--start', type=str, required=True,
                        help='Start of folders')
    parser.add_argument('-qst', '--qstart', type=str, required=False,
                        help='Start of folders for Quincy files')
    parser.add_argument('-e', '--end', type=str, required=True,
                        help='End of files')
    parser.add_argument('-pr', '--prot', type=str, required=True,
                        help='Protein data')
    parser.add_argument('-ex', '--experiment', type=str, required=True,
                        help='Experiment number')
    parser.add_argument('-sa', '--sample', type=str, required=True,
                        help='Sample list')
    parser.add_argument('-co', '--conf', type=str, required=True,
                        help='Configuration file')
    parser.add_argument('-ip', '--imagepath', type=str, required=True,
                        help='Image file')
    parser.add_argument('-d', '--debug', type=str, required=False,
                        help='Image file')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    slide_table = args.slide_table
    ant_dbase = args.ant_dbase
    path = args.path
    pmt_settings = args.pmt
    start = args.start
    end = args.end
    experiment = args.experiment
    conf_file = args.conf
    image_path = args.imagepath
    if args.debug:
        debug = args.debug
    else:
        debug = 0

    prot_data = pd.read_table(args.prot, sep="\t", skiprows=32)

    rppa = RPPA(slide_table, ant_dbase)

    norm_data = rppa.normalize(args.path,start,end,pmt_settings,prot_data,debug)
    #qnorm_data = rppa.norm_data(args.qpath, args.qstart, pmt_settings)
    raw_data = rppa.raw_data(args.path, start, end, pmt_settings)
    flag_data = rppa.flag_data(args.path, start, end, pmt_settings)

    for pmt in pmt_settings:
        print pmt
        norm_data[pmt].to_csv(experiment+"_Norm_"+pmt+".xls", sep="\t", na_rep='NA')
        raw_data[pmt].to_csv(experiment+"_Raw_"+pmt+".xls", sep="\t", na_rep='NA')
        flag_data[pmt].to_csv(experiment+"_Flag_"+pmt+".xls", sep="\t", na_rep='NA')

    for pmt in pmt_settings:
        rppa.cv(experiment, experiment+"_Raw_"+pmt+".xls", pmt)
        rppa.cv(experiment, experiment+"_Norm_"+pmt+".xls", pmt)

    for pmt in pmt_settings:
        print pmt		
    sample_data = pd.read_table(experiment+"_Norm_"+pmt+".xls", sep="\t", index_col=0)
    sample_list = pd.read_table(args.sample, sep="\t", index_col=0)

    sample_table_file = rppa.sample_table(sample_data, sample_list, experiment)

    list1 = [["Ctrl_IgGmix", "0.", "blank.", "Ctrl_GridHP", "empty."], []]

    for pmt in pmt_settings:
        rppa.pi_data(experiment+"_Norm_"+pmt+".xls", sample_table_file,
                     experiment, list1, conf_file,
                     experiment+"_Norm_"+pmt+"_CV_protein_per_antibody.xls")
        rppa.pi_data(experiment+"_Raw_"+pmt+".xls", sample_table_file,
                     experiment, list1, conf_file,
                     experiment+"_Norm_"+pmt+"_CV_protein_per_antibody.xls")
        rppa.pi_data(experiment+"_Flag_"+pmt+".xls", sample_table_file,
                     experiment, list1, conf_file,
                     experiment+"_Norm_"+pmt+"_CV_protein_per_antibody.xls")

    pi_list = []
    mouse_pi_list = []
    # pi_list = ["YL"]
    # mouse_pi_list = [""]
    with open(conf_file) as f:
        while True:
            text1 = f.readline()
            if text1 == "":
                break
            cols = text1.strip().split('\t')
            if cols[0] == "pi_list":
                pi_list_len = len(cols)
                for i in range(1, pi_list_len):
                    pi_list.append(cols[i])
            if cols[0] == "mouse_pi_list":
                mouse_pi_list_len = len(cols)
                for i in range(1, mouse_pi_list_len):
                    mouse_pi_list.append(cols[i])

    rppa.pi_directory(experiment, pmt_settings, pi_list, mouse_pi_list)

    rppa.reports(experiment, pi_list, mouse_pi_list, "")

    # return imagedata, qc_mean_data, qc_sd_data, bio_rep, bio_rep_std
    imagedata = rppa.imagedf(experiment, image_path)

    for pmt in pmt_settings:
        bar_data = rppa.ctrl_bardata(experiment,
                                     experiment+"_Norm_"+pmt+".xls",
                                     conf_file, list1)
        print bar_data[2]
        rppa.combined_plots(bar_data[0], bar_data[1], bar_data[2], bar_data[3],
                            pmt,
                            experiment+"_Norm_"+pmt+"_Barcharts_with_max.pdf",
                            imagedata=imagedata, bars_for_pi=None)
        rppa.combined_plots((bar_data[0]).drop("Slide_Max", axis=1),
                            (bar_data[1]).drop("Slide_Max", axis=1),
                            bar_data[2], bar_data[3],
                            pmt,
                            experiment+"_Norm_"+pmt+"_Barcharts_without_max.pdf",
                            imagedata=imagedata, bars_for_pi=None)

    for pmt in pmt_settings:
        bar_data = rppa.ctrl_bardata(experiment,
                                     experiment+"_Raw_"+pmt+".xls",
                                     conf_file, list1)
        rppa.combined_plots(bar_data[0], bar_data[1], bar_data[2], bar_data[3],
                            pmt,
                            experiment+"_Raw_"+pmt+"_Barcharts_with_max.pdf",
                            imagedata=imagedata, bars_for_pi=None)
        rppa.combined_plots((bar_data[0]).drop("Slide_Max", axis=1),
                            (bar_data[1]).drop("Slide_Max", axis=1),
                            bar_data[2], bar_data[3],
                            pmt,
                            experiment+"_Raw_"+pmt+"_Barcharts_without_max.pdf",
                            imagedata=imagedata, bars_for_pi=None)

    master_rawdata_dic = {}
    master_normdata_dic = {}
    for pmt in pmt_settings:
        data_org = pd.read_table(experiment+"_Raw_"+pmt+".xls",
                                 sep="\t", index_col=0)
        master_rawdata_dic[pmt] = data_org.ix[:, 1:]

        data_org = pd.read_table(experiment+"_Norm_"+pmt+".xls",
                                 sep="\t", index_col=0)
        master_normdata_dic[pmt] = data_org.ix[:, 1:]
        # print (master_normdata_dic[pmt]).iloc[0:1, :]

    pi_bar_data = rppa.ctrl_bardata(experiment,
                                    experiment+"_Norm_"+pmt+".xls",
                                    conf_file, list1, pi="yes",
                                    master_data_dic=master_normdata_dic,
                                    datatype="Norm")
    # print (pi_bar_data[0]).columns
    # print (pi_bar_data[1]).columns
    # print (pi_bar_data[2]).keys()
    # print (pi_bar_data[3]).keys()
    rppa.combined_plots(pi_bar_data[0], pi_bar_data[1],
                        pi_bar_data[2], pi_bar_data[3],
                        pmt, experiment+"_Norm_PI_barcharts_with_max.pdf",
                        imagedata=None, bars_for_pi="yes")
    rppa.combined_plots((pi_bar_data[0]).drop("Slide_Max", axis=1),
                        (pi_bar_data[1]).drop("Slide_Max", axis=1),
                        pi_bar_data[2], pi_bar_data[3],
                        pmt, experiment+"_Norm_PI_barcharts_without_max.pdf",
                        imagedata=None, bars_for_pi="yes")

    pi_bar_data = rppa.ctrl_bardata(experiment,
                                    experiment+"_Raw_"+pmt+".xls",
                                    conf_file, list1, pi="yes",
                                    master_data_dic=master_rawdata_dic,
                                    datatype="Raw")

    rppa.combined_plots(pi_bar_data[0], pi_bar_data[1],
                        pi_bar_data[2], pi_bar_data[3],
                        pmt, experiment+"_Raw_PI_barcharts_with_max.pdf",
                        imagedata=None, bars_for_pi="yes")
    rppa.combined_plots((pi_bar_data[0]).drop("Slide_Max", axis=1),
                        (pi_bar_data[1]).drop("Slide_Max", axis=1),
                        pi_bar_data[2], pi_bar_data[3],
                        pmt, experiment+"_Raw_PI_barcharts_without_max.pdf",
                        imagedata=None, bars_for_pi="yes")

if __name__ == '__main__':
    try:
        main()
    except:
        print "An unknown error occurred.\n"
        raise
