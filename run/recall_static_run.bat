@echo off
start /wait call cmd /k "cd /d D:\DiskANN\x64\Release && compute_groundtruth.exe  --data_type float --dist_fn l2 --base_file D:\DiskANN\build\data\ann_inputs\ann_inputs_learn.fbin --query_file  D:\DiskANN\build\data\ann_inputs\ann_inputs_query.fbin --gt_file D:\DiskANN\build\data\ann_inputs\ann_inputs_query_learn_gt100 --K 100"
start /wait call cmd /k "cd /d D:\DiskANN\x64\Release && build_memory_index.exe  --data_type float --dist_fn l2 --data_path D:\DiskANN\build\data\ann_inputs\ann_inputs_learn.fbin --index_path_prefix D:\DiskANN\build\data\ann_inputs\index_ann_inputs_learn_R128_L100_A1.2 -R 128 -L 100 -T 1 --alpha 1.2"
start /wait call cmd /k "cd /d D:\DiskANN\x64\Release && search_memory_index.exe  --data_type float --dist_fn l2 --index_path_prefix D:\DiskANN\build\data\ann_inputs\index_ann_inputs_learn_R128_L100_A1.2 --query_file D:\DiskANN\build\data\ann_inputs\ann_inputs_query.fbin  --gt_file D:\DiskANN\build\data\ann_inputs\ann_inputs_query_learn_gt100 -K 10 -L 30 -T 1 --result_path D:\DiskANN\build\data\ann_inputs\ann_inputs_static_res"