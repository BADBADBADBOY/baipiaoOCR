def rmdir():
    import os
    import shutil
    remove_dir_list = [".ipynb_checkpoints","__pycache__",".idea"]
    path = './'
    for maindir, subdir, file_name_list in os.walk(path):
        for _dir in subdir:
            re_dir = os.path.join(maindir, _dir)
            for item in remove_dir_list:
                if item in re_dir:
                    shutil.rmtree(re_dir)
    os.system("rm -r ocrCls/new_model_dir/*")
    os.system("rm -r ocrDetect/new_model_dir/*")
    os.system("rm -r ocrRecog/new_model_dir/*")
    os.system("rm -r ./serviceOCRModule")
    os.system("rm -r ./*.jpg")
                    
rmdir()