import os, sys, argparse, re
import numpy as np
import scipy
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as dist
import pandas as pd
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats
%matplotlib inline

def main():

	experiment = "RPPA0038"
	pmt_settings = ["360PMT","380PMT","400PMT","460PMT","500PMT","550PMT"] #550PMT 520PMT 460PMT 400PMT 380PMT 360PMT
	data_types = ["Norm"]

	#pmt_settings = ["460PMT"]
	#data_types = ["Norm"]
	#pmt = "460PMT"
	#data_type = "Raw"

	database = pd.read_table("BCM-RPPA_Core_10-10-2019.xls", sep="\t",index_col=0)

	slidedata_all = pd.read_table(experiment+"_slide_table.xls", sep="\t")
	grouped_slide = slidedata_all.groupby(slidedata_all["Ab_ID"].values)
	slidedata = grouped_slide.first()

	for pmt in pmt_settings:
		for data_type in data_types:

			raw_data = pd.read_table(experiment+"_"+data_type+"_"+pmt+".xls", sep="\t", index_col=0)
			raw_data1 = raw_data.ix[:,1:]
			
			drop_items = ["Ctrl_IgGmix","0.","blank.","Ctrl_GridHP","empty."]

			drop_list = []
			if drop_items is not None:
				for item in (drop_items):
					for i in raw_data1.columns:
						#print drop_items[0],item,i
						if re.match(item, i):
							drop_list.append(i)
							#print i
			data = raw_data1.drop(drop_list, axis=1)

			qc_data = pd.DataFrame(np.nanmean(data, axis=1),index=data.index,columns=["Slide_Mean"])
			qc_data['Slide_Median'] = np.nanmedian(data, axis=1)
			qc_data['Slide_Max'] = np.nanmax(data, axis=1)

			fac = []
			for i in data.columns.values:
				a = i.split('_')
				fac.append((a[0]))

			grouped = data.groupby([fac], axis=1)
			#sample_grouped = self.sampledata_clean.groupby([fac], axis=1)

			data_ctrl = grouped.get_group('Ctrl')
			ctrl_fac = []
			for i in data_ctrl.columns.values:
				a = i.split('_')
				ctrl_fac.append((a[1]))
			grouped_ctrl = data_ctrl.groupby([ctrl_fac], axis=1)

			data_cal = grouped.get_group('Cal')
			cal_fac = []
			for i in data_cal.columns.values:
				a = i.split('_')
				cal_fac.append(a[1])
			grouped_cal = data_cal.groupby([cal_fac], axis=1)

			ctrl_data = {}
			for i in grouped_ctrl.groups:
				#print i
				ctrl_data[i] = grouped_ctrl.get_group(i)

			cal_data = {}
			for i in grouped_cal.groups:
				#print i
				cal_data[i] = grouped_cal.get_group(i)            

			fac = []
			for i in (ctrl_data['Cellmix1']).columns.values:
				a = i.split('_')
				fac.append((a[2]))
			grouped_celllmix = (ctrl_data['Cellmix1']).groupby([fac], axis=1, sort=False)
			cellmix_mean = grouped_celllmix.aggregate(np.mean)
			cellmix_sd = grouped_celllmix.aggregate(np.std)

			r_val1 = {}
			p_val1 = {}
			conc_array = 8.0-(cellmix_mean.columns.values).astype(np.float)#This works perfectly for following 1,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125
			for i in cellmix_mean.index:
				#print i
				ant_exp = (cellmix_mean.ix[i,:]).values
				# if len(ant_exp) == 2:
				#    ant_exp = ant_exp[1]
				cor = scipy.stats.pearsonr(conc_array, ant_exp)
				#print cor[0],cor[1]
				r_val1[i] = cor[0]
				p_val1[i] = cor[1]

			fac = []
			for i in (ctrl_data['Cellmix2']).columns.values:
				a = i.split('_')
				fac.append((a[2]))
			grouped_celllmix = (ctrl_data['Cellmix2']).groupby([fac], axis=1, sort=False)
			cellmix_mean = grouped_celllmix.aggregate(np.mean)
			cellmix_sd = grouped_celllmix.aggregate(np.std)

			r_val2 = {}
			p_val2 = {}
			conc_array = 8.0-(cellmix_mean.columns.values).astype(np.float)#This works perfectly for following 1,0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125
			for i in cellmix_mean.index:
				#print i
				ant_exp = (cellmix_mean.ix[i,:]).values
				# if len(ant_exp) == 2:
				#    ant_exp = ant_exp[1]
				cor = scipy.stats.pearsonr(conc_array, ant_exp)
				#print cor[0],cor[1]
				r_val2[i] = cor[0]
				p_val2[i] = cor[1]
			   
			#qc_data['Ctrl_BSA'] = scipy.stats.nanmean((ctrl_data["BSA1"]), axis = 1)/qc_data["Slide_Median"]
			#qc_data['Ctrl_Chickenovalbumin'] = scipy.stats.nanmean((ctrl_data["Chickenovalbumin1"]), axis = 1)/qc_data["Slide_Median"]
			#qc_data['Ctrl_I-Block'] = scipy.stats.nanmean((ctrl_data["I-Block1"]), axis = 1)/qc_data["Slide_Median"]

			fac = []
			for i in data.columns.values:
				a = re.split(r"(\..$)",i)
				fac.append((a[0]))

			grouped2 = data.groupby([fac], axis=1)

			gr_mean = grouped2.aggregate(np.mean)
			gr_sd = grouped2.aggregate(np.std)
			gr_median = grouped2.aggregate(np.median)
			gr_cv = gr_sd*100/gr_mean


			from pandas.tools.plotting import autocorrelation_plot
			from matplotlib.backends.backend_pdf import PdfPages
			pp = PdfPages(experiment+'_'+data_type+'_'+pmt+'_median_vs_CV_plot_with_cutoff.pdf')
			finite_per = {}
			cutoff = {}
			for i in data.index:
				#for i in [124]:
				#print i
				cv_10_percent = {}
				median_data = gr_median.ix[i,:]
				cv_data = gr_cv.ix[i,:] 
				increment = 100
				fac = []
				end = int(max(gr_median.ix[i,:]))-int((max(gr_median.ix[i,:]))%increment)
				if end == 0:
					end = 100
				#print end
				for signal in range(0, end, increment):
					ll = signal
					ul = signal + increment
					median_range = median_data[(median_data<ul)&(median_data>=ll)]
					if len(median_range) >= 1:
						#print len(median_range)
						cv_10_percent[ul] = (((cv_data[median_range.index]> 25).sum())*100.00/np.float(len(median_range)))
					else:
						cv_10_percent[ul] = "NA"
					#fac.append(signal)

				df_cv10 = pd.DataFrame.from_dict(cv_10_percent, orient='index')
				df_cv10.columns = ["per"]
				try:
					df_cv10_sort = (df_cv10[df_cv10.per!= "NA"]).sort_values(by="per")
				except:
					df_cv10_sort = (df_cv10).sort_values(by="per")
				for k in range(0, len(df_cv10_sort.index)):
					if k <= len(df_cv10_sort.index)-3:
						if df_cv10_sort.ix[df_cv10_sort.index[k],"per"] < 20:
							if ((df_cv10_sort.ix[df_cv10_sort.index[k+1],"per"] < 20) & (df_cv10_sort.ix[df_cv10_sort.index[k+2],"per"] < 20)):
								#print df_cv10_sort.index[k]
								cutoff[i] = df_cv10_sort.index[k]
								break
					else:
						cutoff[i] = "NA"

				if cutoff[i] != "NA":
					finite_per[i] = np.round(np.float64(sum(median_data>cutoff[i]))/np.float64(len(median_data))*100.00,2)
				else:
					finite_per[i] = "NA"

				fig, ax = plt.subplots(1)
				plt.plot(gr_median.ix[i,:],gr_cv.ix[i,:],".")
				ax.set_xlim(1,)
				ax.set_ylim(-1,60)
				ax.set_xscale('log')
				ax.set_ylabel('CV(%) of technical replicates')
				ax.set_xlabel('median intensity of technical replicates')
				if cutoff[i] != "NA":
					ax.vlines(cutoff[i],-1,60,colors='r')
				try:
					ax.set_title(database.ix[int(i),"PI_name"]+"_Ab-ID = "+str(i)+"\nCut-off = "+str(cutoff[i])+
								 ", Samples>cut-off = "+str(finite_per[i])+"%",fontweight='bold')
				except:
					ax.set_title("_Ab-ID = "+str(i)+"\nCut-off = "+str(cutoff[i])+", Samples>cut-off = "+str(finite_per[i])+"%",
								 fontweight='bold')
				pp.savefig(bbox_inches='tight',dpi=50)
				plt.close()
			pp.close()

			qc_data['CV_score'] = ((pd.DataFrame.from_dict(finite_per, orient='index')).convert_objects(convert_numeric=True))*10.00/100.00
			qc_data['Cut-off'] = pd.DataFrame.from_dict(cutoff, orient='index')

			qc_data['cellmix1_r'] = pd.DataFrame.from_dict(r_val1, orient='index')
			qc_data['cellmix1_p'] = pd.DataFrame.from_dict(p_val1, orient='index')
			qc_data['cellmix2_r'] = pd.DataFrame.from_dict(r_val2, orient='index')
			qc_data['cellmix2_p'] = pd.DataFrame.from_dict(p_val2, orient='index')

			def cal_cor(caldf, fac_conc, database):
				rval = {}
				pval = {}
				for i in caldf.index:
					if (database.ix[int(i),"PI_name"]).startswith("p-"):
						ant_exp = (caldf.ix[i,:]).values
						conc_array = (np.array(list(set(fac_conc)))).astype(np.float)
						cor = scipy.stats.pearsonr(conc_array, ant_exp)
						#print cor[0],cor[1]
						rval[i] = cor[0]
						pval[i] = cor[1]
					else:
						rval[i] = np.nan
						pval[i] = np.nan
				return rval, pval

			fac1 = []
			fac2 = []
			fac3 = []
			for i in (cal_data['1']).columns.values:
				a = i.split('_')
				if (a[2]).startswith("HP"):
					fac1.append(i)
					fac2.append(a[2])
					fac3.append((a[2])[2:])
			grouped_cal_ext = ((cal_data['1']).ix[:,fac1]).groupby([fac2], axis=1, sort=False)
			cal_mean = grouped_cal_ext.aggregate(np.mean)      
			calhp1_rval, calhp1_pval = cal_cor(cal_mean, fac3, database)

			fac1 = []
			fac2 = []
			fac3 = []
			for i in (cal_data['2']).columns.values:
				a = i.split('_')
				if (a[2]).startswith("HP"):
					fac1.append(i)
					fac2.append(a[2])
					fac3.append((a[2])[2:])
			grouped_cal_ext = ((cal_data['2']).ix[:,fac1]).groupby([fac2], axis=1, sort=False)
			cal_mean = grouped_cal_ext.aggregate(np.mean)      
			calhp2_rval, calhp2_pval = cal_cor(cal_mean, fac3, database)

			fac1 = []
			fac2 = []
			fac3 = []
			for i in (cal_data['1']).columns.values:
				a = i.split('_')
				if (a[2]).startswith("JC"):
					fac1.append(i)
					fac2.append(a[2])
					fac3.append((a[2])[2:])
			grouped_cal_ext = ((cal_data['1']).ix[:,fac1]).groupby([fac2], axis=1, sort=False)
			cal_mean = grouped_cal_ext.aggregate(np.mean)      
			caljc1_rval, caljc1_pval = cal_cor(cal_mean, fac3, database)

			fac1 = []
			fac2 = []
			fac3 = []
			for i in (cal_data['2']).columns.values:
				a = i.split('_')
				if (a[2]).startswith("JC"):
					fac1.append(i)
					fac2.append(a[2])
					fac3.append((a[2])[2:])
			grouped_cal_ext = ((cal_data['2']).ix[:,fac1]).groupby([fac2], axis=1, sort=False)
			cal_mean = grouped_cal_ext.aggregate(np.mean)      
			caljc2_rval, caljc2_pval = cal_cor(cal_mean, fac3, database)

			qc_data['CalHP1_r'] = pd.DataFrame.from_dict(calhp1_rval, orient='index')
			qc_data['CalHP1_p'] = pd.DataFrame.from_dict(calhp1_pval, orient='index')
			qc_data['CalHP2_r'] = pd.DataFrame.from_dict(calhp2_rval, orient='index')
			qc_data['CalHP2_p'] = pd.DataFrame.from_dict(calhp2_pval, orient='index')
			qc_data['CalJC1_r'] = pd.DataFrame.from_dict(calhp1_rval, orient='index')
			qc_data['CalJC1_p'] = pd.DataFrame.from_dict(calhp1_pval, orient='index')
			qc_data['CalJC2_r'] = pd.DataFrame.from_dict(calhp2_rval, orient='index')
			qc_data['CalJC2_p'] = pd.DataFrame.from_dict(calhp2_pval, orient='index')
			
			
			cellmix_score = {}
			qc_score = {}
			for i in data.index:
				if ((qc_data.ix[i,"cellmix1_p"]) <= 0.05) & ((qc_data.ix[i,"cellmix2_p"]) <= 0.05):
					cellmix_score[i] = max(qc_data.ix[i,"cellmix1_r"],qc_data.ix[i,"cellmix2_r"])*100.00
				elif ((qc_data.ix[i,"cellmix1_p"]) <= 0.05):
					cellmix_score[i] = (qc_data.ix[i,"cellmix1_r"])*100.00
				elif ((qc_data.ix[i,"cellmix2_p"]) <= 0.05):
					cellmix_score[i] = (qc_data.ix[i,"cellmix2_r"])*100.00
				else:
					cellmix_score[i] = 0.0

				qc_score[i] = min([cellmix_score[i],qc_data.ix[i,"CV_score"]])

			qc_data['Cellmix_score'] = (pd.DataFrame.from_dict(cellmix_score, orient='index').convert_objects(convert_numeric=True))*90.00/100.00
			#qc_data['QC_score'] = pd.DataFrame.from_dict(qc_score, orient='index')
			qc_data['QC_score'] = qc_data['Cellmix_score'] + qc_data['CV_score']

			ab_name = {}
			ab_batch = {}
			for i in qc_data.index:
				ab_batch[i] = slidedata.ix[i,"Experiment"]
				if str(i) in database.index.astype(str):
					#print i
					if database.ix[int(i),"Host"] == "Mouse":
						h = "M"
					elif database.ix[int(i),"Host"] == "Rabbit":
						h = "R"
					elif database.ix[int(i),"Host"] == "Goat":
						h = "G"
					else:
						h = "N"

					if database.ix[int(i),"Current_Validation_Status"] == "Validated":
						s = "V"
					elif database.ix[int(i),"Current_Validation_Status"] == "Not_Valid":
						s = "N"
					elif database.ix[int(i),"Current_Validation_Status"] == "Caution":
						s = "C"
					elif database.ix[int(i),"Current_Validation_Status"] == "Progress":
						s = "P"
					elif database.ix[int(i),"Current_Validation_Status"] == "Failed":
						s = "F"
					else:
						s = ""
					ab_name[i] = database.ix[int(i),"PI_name"]+"_"+h+"_"+s
					

			ab_name_column = pd.DataFrame(pd.Series(ab_name))
			ab_batch_column = pd.DataFrame(pd.Series(ab_batch))
			qc_data.insert(0, "AB_name", ab_name_column)
			qc_data.insert(4, "Batch", ab_batch_column)
				  
			qc_table_data = pd.read_table(experiment+"_"+data_type+"_"+pmt+"_quality_table.xls", sep="\t", index_col=0)
			
			qc_data_combined = qc_table_data.join(qc_data.ix[:,4:])
			qc_data_combined.to_csv(experiment+"_"+data_type+"_"+pmt+"_QC_data.xls",sep="\t", na_rep='NA')
			

	for pmt in pmt_settings:

		print pmt
		
		slidedata = slidedata_all[(slidedata_all["Antibody_file"]).str.contains(pmt)]
		
		raw_data_all = pd.read_table(experiment+"_Raw_"+pmt+".xls", sep="\t", index_col=0)
		raw_data1 = raw_data_all.ix[:,1:]
		drop_items = ["Ctrl_IgGmix","0.","blank.","Ctrl_GridHP","empty."]
		drop_list = []
		if drop_items is not None:
			for item in (drop_items):
				for i in raw_data1.columns:
					#print drop_items[0],item,i
					if re.match(item, i):
						drop_list.append(i)
						#print i
		raw_data = raw_data1.drop(drop_list, axis=1)
		fac = []
		for i in raw_data.columns.values:
			a = re.split(r"(\..$)",i)
			fac.append((a[0]))
		grouped2 = raw_data.groupby([fac], axis=1)
		raw_gr_mean = grouped2.aggregate(np.mean)
		raw_gr_sd = grouped2.aggregate(np.std)

		norm_data_all = pd.read_table(experiment+"_Norm_"+pmt+".xls", sep="\t", index_col=0)
		norm_data1 = norm_data_all.ix[:,1:]
		#drop_items = ["Ctrl_IgGmix","0.","blank.","Ctrl_GridHP","empty."]
		drop_list = []
		if drop_items is not None:
			for item in (drop_items):
				for i in norm_data1.columns:
					#print drop_items[0],item,i
					if re.match(item, i):
						drop_list.append(i)
						#print i
		norm_data = norm_data1.drop(drop_list, axis=1)
		fac = []
		for i in norm_data.columns.values:
			a = re.split(r"(\..$)",i)
			fac.append((a[0]))
		grouped2 = norm_data.groupby([fac], axis=1)
		norm_gr_mean = grouped2.aggregate(np.mean)
		norm_gr_mean.replace([np.inf, -np.inf], np.nan)
		norm_gr_sd = grouped2.aggregate(np.std)
		#norm_gr_mean.fillna("NA")

		from matplotlib.backends.backend_pdf import PdfPages
		pp = PdfPages(experiment+'_'+pmt+'_top10_bar_plot.pdf')

		for row in slidedata.index:
			ant_cols = (slidedata.ix[row,5]).split("_")
			abid = (ant_cols[5])[2:]

			neg_cols = (slidedata.ix[row,7]).split("_")
			neg_id = "Ne"+neg_cols[3]+neg_cols[2]

			tot_cols = (slidedata.ix[row,6]).split("_")
			tot_id = "Pr"+tot_cols[3]

			vec_norm_sort = (norm_gr_mean.ix[int(abid),:])
			vec_norm_sort = vec_norm_sort.sort_values(ascending=False)
			top_samples = vec_norm_sort.index[0:10]
			#vec_raw = (raw_data.ix[(abid),:])
			#vec_neg = (raw_data.ix[(neg_id),:])

			top_norm_mean = norm_gr_mean.ix[int(abid),top_samples]
			top_norm_sd = norm_gr_sd.ix[int(abid),top_samples]
			top_raw_mean = raw_gr_mean.ix[(abid),top_samples]
			top_raw_sd = raw_gr_sd.ix[(abid),top_samples]
			top_neg_mean = raw_gr_mean.ix[(neg_id),top_samples]
			top_neg_sd = raw_gr_sd.ix[(neg_id),top_samples]
			"""top_norm_mean = (vec_norm_sort[vec_norm_sort.index[0:10]]).values
			top_raw = (raw_data_sr[norm_data_sort.index[0:10]]).values
			matrix_neg = (neg_data_sr[norm_data_sort.index[0:10]]).values"""

			fig, ax = plt.subplots(1)
			ind = np.arange(0, 2*len(top_norm_mean), 2)
			bar1 = ax.bar(ind, top_norm_mean, width=0.5, label=top_samples, color="r", edgecolor = "none",
											   align='center', log=0)
			ax.errorbar(ind, top_norm_mean,yerr=top_norm_sd, fmt='none',ecolor='black',elinewidth=0.5, capsize=1,capthick=0.5)
			bar2 = ax.bar(ind+0.5, top_raw_mean, width=0.5, label=top_samples, color="y", edgecolor = "none",
											   align='center', log=0)
			ax.errorbar(ind+0.5, top_raw_mean,yerr=top_raw_sd, fmt='none',ecolor='black',elinewidth=0.5, capsize=1,capthick=0.5)
			bar3 = ax.bar(ind+1.0, top_neg_mean, width=0.5, label=top_samples, color="b", edgecolor = "none",
											   align='center', log=0)
			ax.errorbar(ind+1.0, top_neg_mean,yerr=top_neg_sd, fmt='none',ecolor='black',elinewidth=0.5, capsize=1,capthick=0.5)
			ax.legend((bar1[0], bar2[0], bar3[0]), ('Norm', 'Raw', 'Neg'), prop={'size': 5}, loc='best', fancybox=True,
						   framealpha=0.5)
			# ax.set_yscale('log')
			ax.set_xticks(ind+0.5)
			ax.set_xticklabels(top_samples, fontsize=4, rotation=45,ha='right')
			for tick in ax.yaxis.get_major_ticks():
				tick.label.set_fontsize(4)
			ax.set_xlim(-1)
			ax.set_ylim(-1)

			try:
				ax.set_title(database.ix[int(abid),"PI_name"]+"_Ab-ID = "+str(abid),fontweight='bold')
			except:
				ax.set_title("_Ab-ID = "+str(abid), fontweight='bold')

			pp.savefig(bbox_inches='tight',dpi=50)
			plt.close()

		pp.close()
	
if __name__ == '__main__':
    try:
        main()
    except:
        print "An unknown error occurred.\n"
        raise
