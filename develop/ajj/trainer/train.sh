#!/bin/bash

# 1_python base_res50_cos_16_4.py
# 2_python base_res50_crop_JI_16_4.py
# python 2dot5_base_res50_JI_16_4.py
# python 3_res50_BC_16_4.py
# python 4_res50_BC_fp16_16_8.py
# python 4dot5_res50_BC_fp16_8_4.py
python 5_res50_BC_fp16_ignore_8_4.py
python 6_res50_BC_fp16_ignore_step_8_4.py
# python 7_res50_BC_fp16_ignore_step_skfd_8_4.py
python 8_res50_BC_fp16_ignore_step_skfd_hardaug_8_4.py
# python 9_res50_BC_fp16_ignore_step_sgkf_hardaug_4_2_1024.py
# python 10_unet_3plus_BC_fp16_ignore_step_sgkf_hardaug_4_2_512.py