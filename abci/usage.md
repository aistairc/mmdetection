# ABCI option
Login to ABCI, pull docker image:

    $ module load singularitypro/3.7
    $ singularity pull abci/simg/docker://ishitsukah/mmdetection

Run demo batch script:

    $ qsub -g <group> abci/demo.sh

Result files is <bind directory/outputs/demo>.
