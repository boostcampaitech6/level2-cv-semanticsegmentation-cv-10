#!/bin/bash

# 1_python base_res50_cos_16_4.py
# 2_python base_res50_crop_JI_16_4.py
# python 2dot5_base_res50_JI_16_4.py
python 3_base_res50_BC_16_4.py
python 4_base_res50_step_BC_16_4.py
python 5_res50_sgkf_step_BC_16_4.py
python 6_res50_sgkf_ignore_step_BC_16_4.py
python 7_res50_sgkf_ignore_step_fp16_32_8.py
python 8_res50_sgkf_ignore_step_fp16_hardaug_32_8.py
# python 9_res50_sgkf_ignore_step_fp16_hardaug_4_2_1024.py
# python 10_unet_3plus_sgkf_step_fp16_hardaug_4_2_512.py

