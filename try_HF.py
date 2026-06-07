from hf_utils import upload_to_hf

# After training is done:
upload_to_hf(
    repo_name="sam3-medical-segmentation", 
    local_path="/home/ahma/Medical_Segmentation/pipeline/SAM3/sam3_best.pth",
    token=""
)