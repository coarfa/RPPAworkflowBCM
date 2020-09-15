#!/bin/bash
#PBS -N job_rppa
#PBS -o ./
#PBS -e ./
#PBS -l nodes=1:ppn=8
#PBS -l vmem=12gb
#PBS -l walltime=18:00:00
source /etc/profile.d/modules.sh
cd $PBS_O_WORKDIR
module load anaconda/2.1.0
python ~/tools/repos/hub_rppa/RPPA_main.py -sl RPPA0038_slide_table.xls -a /mount/omalley/kimal/Shixia/BCM-RPPA_Core_01-22-2020_for_RPPA0038.xls -p1 RPPA0038_gpr/ \
-p2 360PMT 380PMT 400PMT 460PMT 500PMT 550PMT -st RPPA0038_ \
-qst  RPPA0038_ -e .gpr -pr RPPA0038_sample_list_gpr.txt -ex RPPA0038 -sa RPPA0038_sample_list.txt -co RPPA0038_configuration_file.txt -ip RPPA0038_jpg/
# 360PMT 380PMT 400PMT 460PMT
# 390PMT 340PMT 320PMT 310PMT
