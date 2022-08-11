@echo off
start /wait call compute_groundtruth.bat
start /wait call build_memory_index.bat
start /wait call search_memory_index.bat


