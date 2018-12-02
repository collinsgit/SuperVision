# Requires python 3 with pytorch/CUDA (calling it just 'python' not 'python3')
# ALWAYS run from the /src directory
# Needs MSCOCO 2017

# Crop, Resize, Filter, and Average training/test data

# Running with -h will list all options
python prepareData.py path_to_unzipped_training path_to_unzipped_val --unzipped_annotation_dir path_to_unzipped_annotation

# Run CEDSR (Classified-EDSR) for x4 scale
# To use an existing model, replace 'download' with something like:
# "../models/edsr_baseline_x4-6b446fab.pt" 
# "../experiment/convavg/model_1.pt" 
python main.py "--data_train" "Coco" "--data_test" "Coco" "--use_classification" \
                "--pre_train" \
                "download" \
                "--data_range" \
                "all" \
                "--scale" \
                "4" \
                "--save_models" \
                "--test_every" \
                "100" \
                "--batch_size" \
                "16" \
                "--epochs" \
                "20" \
                "--save" \
                "convavg"
