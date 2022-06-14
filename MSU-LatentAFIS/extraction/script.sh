# bash model_inference.sh 0 joint copied_models/3runs/CM_joint_101_5_1_49.pth 5 CM 1 | tee CMlog1
# bash model_inference.sh 0 joint copied_models/3runs/CM_joint_101_5_2_49.pth 5 CM 2 | tee CMlog2
# bash model_inference.sh 0 joint copied_models/3runs/CM_joint_101_5_3_49.pth 5 CM 3 | tee CMlog3

# bash model_inference.sh 0 joint copied_models/3runs/DP_joint_101_5_1_49.pth 5 DP 1 | tee DPlog1
# bash model_inference.sh 0 joint copied_models/3runs/DP_joint_101_5_2_49.pth 5 DP 2 | tee DPlog2
# bash model_inference.sh 0 joint copied_models/3runs/DP_joint_101_5_3_49.pth 5 DP 3 | tee DPlog3

# bash model_inference.sh 0 joint copied_models/3runs/GB_joint_101_5_1_49.pth 5 GB 1 | tee GBlog1
# bash model_inference.sh 0 joint copied_models/3runs/GB_joint_101_5_2_49.pth 5 GB 2 | tee GBlog2
# bash model_inference.sh 0 joint copied_models/3runs/GB_joint_101_5_3_49.pth 5 GB 3 | tee GBlog3

# bash model_inference.sh 0 joint copied_models/3runs/HS_joint_101_5_1_49.pth 5 HS 1 | tee HSlog1
# bash model_inference.sh 0 joint copied_models/3runs/HS_joint_101_5_2_49.pth 5 HS 2 | tee HSlog2
# bash model_inference.sh 0 joint copied_models/3runs/HS_joint_101_5_3_49.pth 5 HS 3 | tee HSlog3

# bash model_inference.sh 0 joint copied_models/3runs/CM_joint_101_2_1_49.pth 2 CM 1 | tee CM2log1
# bash model_inference.sh 0 joint copied_models/3runs/CM_joint_101_3_1_49.pth 3 CM 1 | tee CM3log1

# bash model_inference.sh 0 joint copied_models/3runs/DP_joint_101_3_1_49.pth 3 DP 1 | tee DP3log1
# bash model_inference.sh 0 joint copied_models/3runs/DP_joint_101_3_2_49.pth 3 DP 2 | tee DP3log2
# bash model_inference.sh 0 joint copied_models/3runs/DP_joint_101_3_3_49.pth 3 DP 3 | tee DP3log3

bash model_inference.sh 0 joint /scratch/additya/resnet_LD15_joint_sp2_30.pth 2 CM 1 1 10 resnet18 | tee CM_resnet_sp2_110_log1
