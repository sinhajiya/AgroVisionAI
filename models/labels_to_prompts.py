import os


def folder_to_prompt(folder_name):
    # replace underscores and junk
    clean = folder_name.replace("___", " ").replace("__", " ").replace("_", " ")
    parts = clean.split()

    crop = parts[0]     
    disease = " ".join(parts[1:]) if len(parts) > 1 else "healthy"

    disease = disease.lower()

    if "healthy" in disease:
        return f"a photo of a healthy {crop.lower()} leaf"
    else:
        return f"a photo of a {crop.lower()} leaf with {disease}"

def load_prompts(data_dir):
    folds = os.listdir(data_dir)
    prompts = [folder_to_prompt(f) for f in folds]
    return prompts

if __name__ == "__main__": 
    data_dir = './dataset/archive(1)/plantvillage dataset/color'

    prompts = load_prompts(data_dir)

    for  p in prompts[:10]:
        print( p)
