import os
import json
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.io import loadmat

hf_data_cache = {}

MAX_SIDE = 2048
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent


def _download_file(url, dst_path):
    if os.path.exists(dst_path):
        return
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_path = dst_path + ".part"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print(f"Downloading: {url}\n  -> {dst_path}")

    wget_path = shutil.which("wget")
    if wget_path is not None:
        cmd = [wget_path, "-O", tmp_path, url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(f"wget failed for {url}\n{stderr}")
    else:
        urllib.request.urlretrieve(url, tmp_path)

    os.replace(tmp_path, dst_path)


def _safe_extract_tar(tar_path, extract_dir):
    with tarfile.open(tar_path, "r:*") as tf:
        base = os.path.abspath(extract_dir)
        for member in tf.getmembers():
            target = os.path.abspath(os.path.join(extract_dir, member.name))
            if not target.startswith(base + os.sep) and target != base:
                raise Exception(f"Unsafe tar path detected: {member.name}")
        tf.extractall(extract_dir)


def _safe_extract_zip(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, "r") as zf:
        base = os.path.abspath(extract_dir)
        for name in zf.namelist():
            target = os.path.abspath(os.path.join(extract_dir, name))
            if not target.startswith(base + os.sep) and target != base:
                raise Exception(f"Unsafe zip path detected: {name}")
        zf.extractall(extract_dir)


def _ensure_cub_data(data_folder):
    os.makedirs(data_folder, exist_ok=True)
    extracted_root = os.path.join(data_folder, "CUB_200_2011")
    if os.path.isdir(extracted_root):
        return

    tgz_path = os.path.join(data_folder, "CUB_200_2011.tgz")
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
    _download_file(url, tgz_path)
    print(f"Extracting {tgz_path} -> {data_folder}")
    _safe_extract_tar(tgz_path, data_folder)


def _ensure_flowers_data(data_folder):
    os.makedirs(data_folder, exist_ok=True)
    jpg_dir = os.path.join(data_folder, "jpg")
    imagelabels = os.path.join(data_folder, "imagelabels.mat")
    setid = os.path.join(data_folder, "setid.mat")
    readme = os.path.join(data_folder, "README.txt")

    files = [
        ("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz", os.path.join(data_folder, "102flowers.tgz")),
        ("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat", imagelabels),
        ("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat", setid),
        ("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/README.txt", readme),
    ]

    for url, path in files:
        if not os.path.exists(path):
            _download_file(url, path)

    if not os.path.isdir(jpg_dir):
        tgz_path = os.path.join(data_folder, "102flowers.tgz")
        print(f"Extracting {tgz_path} -> {data_folder}")
        _safe_extract_tar(tgz_path, data_folder)


def _ensure_okvqa_data(data_folder):
    os.makedirs(data_folder, exist_ok=True)

    targets = [
        (
            "https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip",
            os.path.join(data_folder, "mscoco_train2014_annotations.json.zip"),
            os.path.join(data_folder, "mscoco_train2014_annotations.json"),
        ),
        (
            "https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip",
            os.path.join(data_folder, "mscoco_val2014_annotations.json.zip"),
            os.path.join(data_folder, "mscoco_val2014_annotations.json"),
        ),
        (
            "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip",
            os.path.join(data_folder, "OpenEnded_mscoco_train2014_questions.json.zip"),
            os.path.join(data_folder, "OpenEnded_mscoco_train2014_questions.json"),
        ),
        (
            "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip",
            os.path.join(data_folder, "OpenEnded_mscoco_val2014_questions.json.zip"),
            os.path.join(data_folder, "OpenEnded_mscoco_val2014_questions.json"),
        ),
        (
            "http://images.cocodataset.org/zips/train2014.zip",
            os.path.join(data_folder, "train2014.zip"),
            os.path.join(data_folder, "train2014"),
        ),
        (
            "http://images.cocodataset.org/zips/val2014.zip",
            os.path.join(data_folder, "val2014.zip"),
            os.path.join(data_folder, "val2014"),
        ),
    ]

    for url, archive_path, extracted_target in targets:
        if os.path.exists(extracted_target):
            continue
        _download_file(url, archive_path)
        print(f"Extracting {archive_path} -> {data_folder}")
        _safe_extract_zip(archive_path, data_folder)


def _ensure_vizwiz_data(data_folder):
    os.makedirs(data_folder, exist_ok=True)

    train_dir = os.path.join(data_folder, "train")
    val_dir = os.path.join(data_folder, "val")
    train_json = os.path.join(data_folder, "train.json")
    val_json = os.path.join(data_folder, "val.json")

    train_zip = os.path.join(data_folder, "train.zip")
    val_zip = os.path.join(data_folder, "val.zip")
    anno_zip = os.path.join(data_folder, "Annotations.zip")

    if not os.path.isdir(train_dir):
        _download_file("https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip", train_zip)
        print(f"Extracting {train_zip} -> {data_folder}")
        _safe_extract_zip(train_zip, data_folder)

    if not os.path.isdir(val_dir):
        _download_file("https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip", val_zip)
        print(f"Extracting {val_zip} -> {data_folder}")
        _safe_extract_zip(val_zip, data_folder)

    if not (os.path.exists(train_json) and os.path.exists(val_json)):
        _download_file("https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip", anno_zip)
        print(f"Extracting {anno_zip} -> {data_folder}")
        _safe_extract_zip(anno_zip, data_folder)

        anno_dir = os.path.join(data_folder, "Annotations")
        if os.path.isdir(anno_dir):
            for fname in os.listdir(anno_dir):
                src = os.path.join(anno_dir, fname)
                dst = os.path.join(data_folder, fname)
                if os.path.isfile(src):
                    if os.path.exists(dst):
                        os.remove(dst)
                    shutil.move(src, dst)
            try:
                os.rmdir(anno_dir)
            except OSError:
                pass


def _ensure_dataset_available(dataset_name, data_folder):
    if dataset_name == "cub":
        _ensure_cub_data(data_folder)
    elif dataset_name == "flowers":
        _ensure_flowers_data(data_folder)
    elif dataset_name == "okvqa":
        _ensure_okvqa_data(data_folder)
    elif dataset_name == "vizwiz":
        _ensure_vizwiz_data(data_folder)


def load_rgb(path, max_side=MAX_SIDE):

    global hf_data_cache

    if isinstance(path, Image.Image):
        img = path.convert("RGB").copy()
    elif path.startswith("huggingface:"):
        clean_path = path.split("huggingface:")[1]
        repo_id = "/".join(clean_path.split("/")[:2])
        subset = clean_path.split("/")[2]
        split = clean_path.split("/")[3]
        index = int(clean_path.split("/")[4])
        key = "/".join(clean_path.split("/")[5:])
        if repo_id not in hf_data_cache or subset not in hf_data_cache[repo_id] or split not in hf_data_cache[repo_id][subset]:
            from datasets import load_dataset
            ds = load_dataset(repo_id, subset, split=split)
            if repo_id not in hf_data_cache:
                hf_data_cache[repo_id] = {}
            if subset not in hf_data_cache[repo_id]:
                hf_data_cache[repo_id][subset] = {}
            hf_data_cache[repo_id][subset][split] = ds
        img = hf_data_cache[repo_id][subset][split][index][key].convert("RGB").copy()
    else:
        with Image.open(path) as img_:
            img = img_.convert("RGB").copy()
    w, h = img.size
    s = max(w, h)
    if s > max_side:
        scale = max_side / s
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return img
def safe_open_image(path, max_side=MAX_SIDE):
    try:
        return load_rgb(path, max_side=max_side)
    except Exception:
        return None

def resize(img, max_pixels=MAX_SIDE * MAX_SIDE):
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    return img

class Dataset:
    def __init__(self, name=None, task=None, split='test', ids=None, imgpaths=None, questions=None, answers=None, options=None, embedding_dir=None):
        self.name = name
        self.task = task
        self.split = split
        self.ids = ids if ids is not None else []
        self.imgpaths = imgpaths if imgpaths is not None else []
        self.questions = questions if questions is not None else []
        self.answers = answers if answers is not None else []
        self.options = options
        if embedding_dir is None and self.name is not None:
            embedding_dir = str(REPO_ROOT / "embeddings" / self.name)
        self.embedding_dir = embedding_dir
        self.text_embeddings = None
        self.image_embeddings = None
        assert len(self.ids) == len(self.imgpaths) == len(self.questions) == len(self.answers)

    def _default_embedding_path(self, embedding_type):
        if self.embedding_dir is None:
            return None
        if embedding_type not in {"image", "text"}:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")
        os.makedirs(self.embedding_dir, exist_ok=True)
        return os.path.join(self.embedding_dir, f"{self.split}_{embedding_type}_embeddings.pt")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                {
                    "id": self.ids[i],
                    "imgpath": self.imgpaths[i],
                    "question": self.questions[i],
                    "answer": self.answers[i],
                }
                for i in range(*index.indices(len(self)))
            ]
        elif isinstance(index, int):
            return {
                "id": self.ids[index],
                "imgpath": self.imgpaths[index],
                "question": self.questions[index],
                "answer": self.answers[index],
            }
        else:
            raise TypeError("Invalid argument type.")
    
    def get_image(self, index, max_side=MAX_SIDE):
        item = self[index]
        img = safe_open_image(item['imgpath'], max_side=max_side)
        return img

    def get_image_embeddings(self, model, processor, batch_size=32, max_side=MAX_SIDE, save_path=None):
        if save_path is None:
            save_path = self._default_embedding_path("image")
        if save_path is not None and os.path.exists(save_path):
            all_embeddings = torch.load(save_path)
        else:
            all_embeddings = []
            for i in tqdm(range(0, len(self), batch_size), desc="Computing image embeddings"):
                batch_imgs = []
                for j in range(i, min(i + batch_size, len(self))):
                    img = self.get_image(j, max_side=max_side)
                    if img is None:
                        raise ValueError(f"Failed to load image: {self.imgpaths[j]}")
                    batch_imgs.append(img)
                inputs = processor(images=batch_imgs, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    embeddings = model.get_image_features(**inputs)
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings.cpu())
            all_embeddings = torch.cat(all_embeddings, dim=0)
            if save_path is not None:
                torch.save(all_embeddings, save_path)
        return all_embeddings

    def get_text_embeddings(self, model, processor, batch_size=32, save_path=None):
        if save_path is None:
            save_path = self._default_embedding_path("text")
        if save_path is not None and os.path.exists(save_path):
            all_embeddings = torch.load(save_path)
        else:
            all_embeddings = []
            for i in tqdm(range(0, len(self), batch_size), desc="Computing text embeddings"):
                batch_texts = []
                for j in range(i, min(i + batch_size, len(self))):
                    question = self.questions[j]
                    batch_texts.append(question)
                inputs = processor(text=batch_texts, return_tensors="pt", padding=True).to(model.device)
                with torch.no_grad():
                    # embeddings = model.get_text_features(**inputs)
                    embeddings = model.get_text_features(inputs['input_ids'][:, :77])
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings.cpu())
            all_embeddings = torch.cat(all_embeddings, dim=0)
            if save_path is not None:
                torch.save(all_embeddings, save_path)
        return all_embeddings

def load_dataset(dataset_name, data_folder=None, split="test"):
    """
    Load a dataset by name.
    
    Args:
        dataset_name: Name of the dataset ('okvqa', 'vizwiz', 'cub', 'flowers')
        data_folder: Path to the data folder. If None, uses default path.
        split: Dataset split ('train' or 'test')
    
    Returns:
        Dataset object
    """
    dataset_loaders = {
        'okvqa': load_okvqa,
        'vizwiz': load_vizwiz,
        'cub': load_cub,
        'flowers': load_flowers,
    }
    
    default_folders = {
        'okvqa': str(REPO_ROOT / "data" / "okvqa"),
        'vizwiz': str(REPO_ROOT / "data" / "vizwiz"),
        'cub': str(REPO_ROOT / "data" / "cub"),
        'flowers': str(REPO_ROOT / "data" / "flowers"),
    }
    
    if dataset_name not in dataset_loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(dataset_loaders.keys())}")
    
    if data_folder is None:
        data_folder = default_folders[dataset_name]

    _ensure_dataset_available(dataset_name, data_folder)
    
    return dataset_loaders[dataset_name](data_folder=data_folder, split=split)

def load_okvqa(data_folder = "../data/okvqa", split="test"):

    task = "vqa"
    if split == "test":
        anno_file = os.path.join(data_folder, "mscoco_val2014_annotations.json")
        ques_file = os.path.join(data_folder, "OpenEnded_mscoco_val2014_questions.json")
        image_folder = os.path.join(data_folder, "val2014")
        fname_prefix = "COCO_val2014_"
    else:
        anno_file = os.path.join(data_folder, "mscoco_train2014_annotations.json")
        ques_file = os.path.join(data_folder, "OpenEnded_mscoco_train2014_questions.json")
        image_folder = os.path.join(data_folder, "train2014")
        fname_prefix = "COCO_train2014_"

    with open(anno_file, 'r') as f:
        annotations = json.load(f)['annotations']

    with open(ques_file, 'r') as f:
        questions_data = json.load(f)['questions']

    qid2question = {q['question_id']: q['question'] for q in questions_data}

    ids = []
    imgpaths = []
    questions = []
    answers = []

    for anno in annotations:
        qid = anno['question_id']
        image_id = anno['image_id']
        answer = anno['answers'][0]['answer']
        question = qid2question[qid]

        imgpath = os.path.join(image_folder, f"{fname_prefix}{image_id:012d}.jpg")

        ids.append(str(qid))
        imgpaths.append(imgpath)
        questions.append(question)
        answers.append(answer)

    return Dataset(
        name="okvqa",
        task=task,
        split=split,
        ids=ids,
        imgpaths=imgpaths,
        questions=questions,
        answers=answers,
    )

def load_vizwiz(data_folder = "../data/vizwiz", split="test"):
    task = "vqa"

    if split == "test":
        anno_file = os.path.join(data_folder, "val.json")
        image_folder = os.path.join(data_folder, "val")
    else:
        anno_file = os.path.join(data_folder, "train.json")
        image_folder = os.path.join(data_folder, "train")

    with open(anno_file, 'r') as f:
        data = json.load(f)

    ids = []
    imgpaths = []
    questions = []
    answers = []

    for item in data:
        image_name = item['image']
        # question = item['question']
        # add unanswerable option to question
        question = item['question'] + " (If the question is unanswerable based on the image, please answer 'unanswerable'.)"
        
        # VizWiz may have multiple answers or unanswerable questions
        if item['answerable'] == 1 and 'answers' in item and len(item['answers']) > 0:
            answer = item['answers'][0]['answer']
        # elif item['answerable'] == 1 and 'answer' in item:
        #     answer = item['answer']
        else:
            answer = "unanswerable"
        
        imgpath = os.path.join(image_folder, image_name)
        
        ids.append(str(item.get('image', image_name)))
        imgpaths.append(imgpath)
        questions.append(question)
        answers.append(answer)

    return Dataset(
        name="vizwiz",
        task=task,
        split=split,
        ids=ids,
        imgpaths=imgpaths,
        questions=questions,
        answers=answers,
    )

def load_cub(data_folder = "../data/cub", split="test"):

    task = "cls"

    image_folder = os.path.join(data_folder, "CUB_200_2011/images")

    train_test_split = np.loadtxt(os.path.join(data_folder, 'CUB_200_2011/train_test_split.txt'), dtype=int)
    train_indices = train_test_split[np.where(train_test_split[:, 1] == 1)][:, 0]
    test_indices = train_test_split[np.where(train_test_split[:, 1] == 0)][:, 0]

    idx2imgpath = np.loadtxt(os.path.join(data_folder, 'CUB_200_2011/images.txt'), dtype=str)
    idx2imgpath = {int(line[0]): line[1] for line in idx2imgpath}
    id2classname = np.loadtxt(os.path.join(data_folder, 'CUB_200_2011/classes.txt'), dtype=str)
    id2classname = {int(line[0]): '.'.join(line[1].split('.')[1:]) for line in id2classname}

    imageidx2classid = np.loadtxt(os.path.join(data_folder, 'CUB_200_2011/image_class_labels.txt'), dtype=int)
    imageidx2classid = {line[0]: line[1] for line in imageidx2classid}

    if split == "train":
        selected_indices = train_indices
    else:
        selected_indices = test_indices

    question = [f"What is the category of the bird in this image?" for _ in range(len(selected_indices))]
    answer = [id2classname[imageidx2classid[idx]] for idx in selected_indices]
    imgpaths = [os.path.join(image_folder, idx2imgpath[idx]) for idx in selected_indices]
    ids = [str(idx) for idx in selected_indices]

    return Dataset(
        name="cub",
        task=task,
        split=split,
        ids=ids,
        imgpaths=imgpaths,
        questions=question,
        answers=answer,
        options=list(id2classname.values())
    )

def load_flowers(data_folder = "../data/flowers", split="test"):

    task = "cls"

    image_folder = os.path.join(data_folder, "jpg")

    setid = loadmat(os.path.join(data_folder, 'setid.mat'))
    train_indices = setid['trnid'][0]
    val_indices = setid['valid'][0]
    test_indices = setid['tstid'][0]

    min_index = min(train_indices.min(), val_indices.min(), test_indices.min())
    max_index = max(train_indices.max(), val_indices.max(), test_indices.max())

    cat_to_name = {"21": "fire lily", "3": "canterbury bells", "45": "bolero deep blue", "1": "pink primrose", "34": "mexican aster", "27": "prince of wales feathers", "7": "moon orchid", "16": "globe-flower", "25": "grape hyacinth", "26": "corn poppy", "79": "toad lily", "39": "siam tulip", "24": "red ginger", "67": "spring crocus", "35": "alpine sea holly", "32": "garden phlox", "10": "globe thistle", "6": "tiger lily", "93": "ball moss", "33": "love in the mist", "9": "monkshood", "102": "blackberry lily", "14": "spear thistle", "19": "balloon flower", "100": "blanket flower", "13": "king protea", "49": "oxeye daisy", "15": "yellow iris", "61": "cautleya spicata", "31": "carnation", "64": "silverbush", "68": "bearded iris", "63": "black-eyed susan", "69": "windflower", "62": "japanese anemone", "20": "giant white arum lily", "38": "great masterwort", "4": "sweet pea", "86": "tree mallow", "101": "trumpet creeper", "42": "daffodil", "22": "pincushion flower", "2": "hard-leaved pocket orchid", "54": "sunflower", "66": "osteospermum", "70": "tree poppy", "85": "desert-rose", "99": "bromelia", "87": "magnolia", "5": "english marigold", "92": "bee balm", "28": "stemless gentian", "97": "mallow", "57": "gaura", "40": "lenten rose", "47": "marigold", "59": "orange dahlia", "48": "buttercup", "55": "pelargonium", "36": "ruby-lipped cattleya", "91": "hippeastrum", "29": "artichoke", "71": "gazania", "90": "canna lily", "18": "peruvian lily", "98": "mexican petunia", "8": "bird of paradise", "30": "sweet william", "17": "purple coneflower", "52": "wild pansy", "84": "columbine", "12": "colt's foot", "11": "snapdragon", "96": "camellia", "23": "fritillary", "50": "common dandelion", "44": "poinsettia", "53": "primula", "72": "azalea", "65": "californian poppy", "80": "anthurium", "76": "morning glory", "37": "cape flower", "56": "bishop of llandaff", "60": "pink-yellow dahlia", "82": "clematis", "58": "geranium", "75": "thorn apple", "41": "barbeton daisy", "95": "bougainvillea", "43": "sword lily", "83": "hibiscus", "78": "lotus lotus", "88": "cyclamen", "94": "foxglove", "81": "frangipani", "74": "rose", "89": "watercress", "73": "water lily", "46": "wallflower", "77": "passion flower", "51": "petunia"}

    imagelabels = loadmat(os.path.join(data_folder, 'imagelabels.mat'))['labels'][0]
    idx2classname = {i: cat_to_name[str(imagelabels[i - 1])] for i in range(min_index, max_index + 1)}

    if split == "train":
        selected_indices = np.concatenate([train_indices, val_indices])
    else:
        selected_indices = test_indices

    question = [f"What is the category of the flower in this image?" for _ in range(len(selected_indices))]
    answer = [idx2classname[idx] for idx in selected_indices]
    imgpaths = [os.path.join(image_folder, f"image_{idx:05d}.jpg") for idx in selected_indices]
    ids = [str(idx) for idx in selected_indices]

    options = sorted(cat_to_name.items(), key=lambda x: int(x[0]))
    options = [v for k, v in options]

    return Dataset(
        name="flowers",
        task=task,
        split=split,
        ids=ids,
        imgpaths=imgpaths,
        questions=question,
        answers=answer,
        options=options,
    )