@echo off
start /wait call cmd /k "cd /d D:\DiskANN\x64\Release && search_memory_index.exe  --data_type float --dist_fn l2 --query_file  D:\DiskANN\build\data\ann_inputs\ann_inputs_query.fbin  --gt_file  D:\DiskANN\build\data\ann_inputs\ann_inputs_query_learn_dynamic_gt100 -K 10 -L 30 -T 4 --result_path  D:\DiskANN\build\data\ann_inputs\ann_inputs_dynamic_res --dynamic true --tags true --static_tags true --max_points 1000000 --qps 10 --insert_percentage 0.3 --build_percentage 1 --delete_percentage 0"
start /wait call cmd /k "cd /d D:\DiskANN\x64\Release && search_memory_index.exe  --data_type float --dist_fn l2 --query_file  D:\DiskANN\build\data\ann_inputs\ann_inputs_query.fbin  --gt_file  D:\DiskANN\build\data\ann_inputs\ann_inputs_query_learn_dynamic_gt100 -K 10 -L 30 -T 4 --result_path  D:\DiskANN\build\data\ann_inputs\ann_inputs_dynamic_res --dynamic true --tags true --static_tags false --max_points 1000000 --qps 10 --insert_percentage 0.3 --build_percentage 0.7 --delete_percentage 0"