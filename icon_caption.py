# Libraries
from typing import Tuple, List, Union
import easyocr
from paddleocr import PaddleOCR

from PIL import Image
import torch 
import numpy as np
import json
import re

import os
import time

from torchvision.transforms import ToPILImage
import cv2

from torchvision.ops import box_convert
from util.box_annotator import BoxAnnotator 
import supervision as sv
import io
import base64

import torchvision.transforms as transforms
from torchvision.models import resnet50
import faiss
import pickle


def get_xywh(input):
    x, y, w, h = input[0][0], input[0][1], input[2][0] - input[0][0], input[2][1] - input[0][1]
    x, y, w, h = int(x), int(y), int(w), int(h)
    return x, y, w, h

def get_xyxy(input):
    x, y, xp, yp = input[0][0], input[0][1], input[2][0], input[2][1]
    x, y, xp, yp = int(x), int(y), int(xp), int(yp)
    return x, y, xp, yp
    
def check_ocr_box(image_source: Union[str, Image.Image],
                  display_img = True, 
                  output_bb_format='xywh', 
                  goal_filtering=None, 
                  easyocr_args=None, 
                  use_paddleocr=False,
                  paddle_ocr=None,
                  reader=None):
    
    if isinstance(image_source, str):
        image_source = Image.open(image_source)
    if image_source.mode == 'RGBA':
        # Convert RGBA to RGB to avoid alpha channel issues
        image_source = image_source.convert('RGB')
    image_np = np.array(image_source)
    w, h = image_source.size
    if use_paddleocr:
        if easyocr_args is None:
            text_threshold = 0.5
        else:
            text_threshold = easyocr_args['text_threshold']
        result = paddle_ocr.ocr(image_np, cls=False)[0]
        coord = [item[0] for item in result if item[1][1] > text_threshold]
        text = [item[1][0] for item in result if item[1][1] > text_threshold]
    else:  # EasyOCR
        if easyocr_args is None:
            easyocr_args = {}
        result = reader.readtext(image_np, **easyocr_args)
        coord = [item[0] for item in result]
        text = [item[1] for item in result]
    if display_img:
        opencv_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        bb = []
        for item in coord:
            x, y, a, b = get_xywh(item)
            bb.append((x, y, a, b))
            cv2.rectangle(opencv_img, (x, y), (x+a, y+b), (0, 255, 0), 2)
        #  matplotlib expects RGB
        plt.imshow(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))
    else:
        if output_bb_format == 'xywh':
            bb = [get_xywh(item) for item in coord]
        elif output_bb_format == 'xyxy':
            bb = [get_xyxy(item) for item in coord]
    return (text, bb), goal_filtering


def predict_yolo(model, image, box_threshold, imgsz, scale_img, iou_threshold=0.7):
    """ Use huggingface model to replace the original model
    """
    # model = model['model']
    if scale_img:
        result = model.predict(
        source=image,
        conf=box_threshold,
        imgsz=imgsz,
        iou=iou_threshold, # default 0.7
        )
    else:
        result = model.predict(
        source=image,
        conf=box_threshold,
        iou=iou_threshold, # default 0.7
        )
    boxes = result[0].boxes.xyxy#.tolist() # in pixel space
    conf = result[0].boxes.conf
    phrases = [str(i) for i in range(len(boxes))]

    return boxes, conf, phrases


def yolo_icon_box_detection(
    image_source,
    model ,
    BOX_TRESHOLD=0.05,
    imgsz=640,
    iou_threshold=0.1,
    scale_img=True):
    
    if isinstance(image_source, str):
        image_source = Image.open(image_source)
    image_source = image_source.convert("RGB") # for CLIP
    w, h = image_source.size
    if not imgsz:
        imgsz = (h, w)
    # print('image size:', w, h)
    xyxy, logits, phrases = predict_yolo(model=model, image=image_source, box_threshold=BOX_TRESHOLD, imgsz=imgsz, scale_img=scale_img, iou_threshold=0.1)
    xyxy = xyxy / torch.Tensor([w, h, w, h]).to(xyxy.device)
    phrases = [str(i) for i in range(len(phrases))]
    
    return xyxy, logits, phrases


def int_box_area(box, w, h):
    x1, y1, x2, y2 = box
    int_box = [int(x1*w), int(y1*h), int(x2*w), int(y2*h)]
    area = (int_box[2] - int_box[0]) * (int_box[3] - int_box[1])
    return area

def remove_overlap_new(boxes, iou_threshold, ocr_bbox=None):
    '''
    ocr_bbox format: [{'type': 'text', 'bbox':[x,y], 'interactivity':False, 'content':str }, ...]
    boxes format: [{'type': 'icon', 'bbox':[x,y], 'interactivity':True, 'content':None }, ...]

    '''
    assert ocr_bbox is None or isinstance(ocr_bbox, List)

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def intersection_area(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    def IoU(box1, box2):
        intersection = intersection_area(box1, box2)
        union = box_area(box1) + box_area(box2) - intersection + 1e-6
        if box_area(box1) > 0 and box_area(box2) > 0:
            ratio1 = intersection / box_area(box1)
            ratio2 = intersection / box_area(box2)
        else:
            ratio1, ratio2 = 0, 0
        return max(intersection / union, ratio1, ratio2)

    def is_inside(box1, box2):
        # return box1[0] >= box2[0] and box1[1] >= box2[1] and box1[2] <= box2[2] and box1[3] <= box2[3]
        intersection = intersection_area(box1, box2)
        ratio1 = intersection / box_area(box1)
        return ratio1 > 0.80

    # boxes = boxes.tolist()
    filtered_boxes = []
    if ocr_bbox:
        filtered_boxes.extend(ocr_bbox)
    # print('ocr_bbox!!!', ocr_bbox)
    for i, box1_elem in enumerate(boxes):
        box1 = box1_elem['bbox']
        is_valid_box = True
        for j, box2_elem in enumerate(boxes):
            # keep the smaller box
            box2 = box2_elem['bbox']
            if i != j and IoU(box1, box2) > iou_threshold and box_area(box1) > box_area(box2):
                is_valid_box = False
                break
        if is_valid_box:
            if ocr_bbox:
                # keep yolo boxes + prioritize ocr label
                box_added = False
                ocr_labels = ''
                for box3_elem in ocr_bbox:
                    if not box_added:
                        box3 = box3_elem['bbox']
                        if is_inside(box3, box1): # ocr inside icon
                            # box_added = True
                            # delete the box3_elem from ocr_bbox
                            try:
                                # gather all ocr labels
                                ocr_labels += box3_elem['content'] + ' '
                                filtered_boxes.remove(box3_elem)
                            except:
                                continue
                            # break
                        elif is_inside(box1, box3): # icon inside ocr, don't added this icon box, no need to check other ocr bbox bc no overlap between ocr bbox, icon can only be in one ocr box
                            box_added = True
                            break
                        else:
                            continue
                if not box_added:
                    if ocr_labels:
                        filtered_boxes.append({'type': 'icon', 'bbox': box1_elem['bbox'], 'interactivity': True, 'content': ocr_labels, 'source':'box_yolo_content_ocr'})
                    else:
                        filtered_boxes.append({'type': 'icon', 'bbox': box1_elem['bbox'], 'interactivity': True, 'content': None, 'source':'box_yolo_content_yolo'})
            else:
                filtered_boxes.append(box1)
    return filtered_boxes # torch.tensor(filtered_boxes)

def merge_ocr_yolo_boxes(
    screenshot_path,
    ocr_bbox,
    ocr_text,
    xyxy,
    iou_threshold=0.1):
    
    image_source = Image.open(screenshot_path)
    image_source = image_source.convert("RGB")
    w, h = image_source.size
    if ocr_bbox:
        ocr_bbox = torch.tensor(ocr_bbox) / torch.Tensor([w, h, w, h])
        ocr_bbox=ocr_bbox.tolist()
    else:
        print('no ocr bbox!!!')
        ocr_bbox = None
    
    ocr_bbox_elem = [{'type': 'text', 'bbox':box, 'interactivity':False, 'content':txt, 'source': 'box_ocr_content_ocr'} for box, txt in zip(ocr_bbox, ocr_text) if int_box_area(box, w, h) > 0] 
    xyxy_elem = [{'type': 'icon', 'bbox':box, 'interactivity':True, 'content':None} for box in xyxy.tolist() if int_box_area(box, w, h) > 0]
    filtered_boxes = remove_overlap_new(boxes=xyxy_elem, iou_threshold=iou_threshold, ocr_bbox=ocr_bbox_elem)
    
    # sort the filtered_boxes so that the one with 'content': None is at the end, and get the index of the first 'content': None
    filtered_boxes_elem = sorted(filtered_boxes, key=lambda x: x['content'] is None)
    # get the index of the first 'content': None
    starting_idx = next((i for i, box in enumerate(filtered_boxes_elem) if box['content'] is None), -1)
    filtered_boxes = torch.tensor([box['bbox'] for box in filtered_boxes_elem])
    print('len(filtered_boxes):', len(filtered_boxes), starting_idx)

    return starting_idx, filtered_boxes, filtered_boxes_elem

def get_cropped_icons_images(
    starting_idx,
    filtered_boxes,
    image_source):
    
    image_source = Image.open(image_source)
    image_source = image_source.convert("RGB")
    image_source = np.asarray(image_source)
    to_pil = ToPILImage()
    if starting_idx:
        non_ocr_boxes = filtered_boxes[starting_idx:]
    else:
        non_ocr_boxes = filtered_boxes
    croped_pil_image = []
    for i, coord in enumerate(non_ocr_boxes):
        try:
            xmin, xmax = int(coord[0]*image_source.shape[1]), int(coord[2]*image_source.shape[1])
            ymin, ymax = int(coord[1]*image_source.shape[0]), int(coord[3]*image_source.shape[0])
            cropped_image = image_source[ymin:ymax, xmin:xmax, :]
            cropped_image = cv2.resize(cropped_image, (64, 64))
            croped_pil_image.append(to_pil(cropped_image))
        except:
            continue

    return croped_pil_image

def get_icons_caption(
    model,
    processor,
    croped_pil_image,
    batch_size):
    
    prompt = "<CAPTION>"
    time1 = time.time()
    device = model.device
    generated_texts = []
    with torch.inference_mode():  # Optimize memory usage
        for i in range(0, len(croped_pil_image), batch_size):
            start = time.time()
            batch = croped_pil_image[i:i+batch_size]
            inputs = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt").to(device=device)
            generated_ids = model.generate(input_ids=inputs["input_ids"],pixel_values=inputs["pixel_values"],max_new_tokens=20,num_beams=1, do_sample=False)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            generated_text = [gen.strip() for gen in generated_text]
            generated_texts.extend(generated_text)        
    print('Time to get parsed content:', time.time()-time1)

    return generated_texts

def annotate(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str], text_scale: float, 
             text_padding=5, text_thickness=2, thickness=3) -> np.ndarray:
    """    
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates. in cxcywh format, pixel scale
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.
    text_scale (float): The scale of the text to be displayed. 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [f"{phrase}" for phrase in range(boxes.shape[0])]

    box_annotator = BoxAnnotator(text_scale=text_scale, text_padding=text_padding,text_thickness=text_thickness,thickness=thickness) # 0.8 for mobile/web, 0.3 for desktop # 0.4 for mind2web
    annotated_frame = image_source.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels, image_size=(w,h))

    label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh)}
    return annotated_frame, label_coordinates

    
def draw_annotate_boxes(
    image_source,
    filtered_boxes_elem,
    filtered_boxes,
    ocr_text,
    parsed_content_icon,
    logits,
    phrases,
    output_coord_in_ratio=True
    ):
    
    image_source = Image.open(image_source)
    image_source = image_source.convert("RGB")
    width, height = image_source.size
    w, h = width, height
    image_source = np.asarray(image_source)
    
    ocr_text = [f"Text Box ID {i}: {txt}" for i, txt in enumerate(ocr_text)]
    icon_start = len(ocr_text)
    parsed_content_icon_ls = []
    # fill the filtered_boxes_elem None content wth parsed_content_icon in order
    for i, box in enumerate(filtered_boxes_elem):
        if box['content'] is None:
            box['content'] = parsed_content_icon.pop(0)
    for i, txt in enumerate(parsed_content_icon):
        parsed_content_icon_ls.append(f"Icon Box ID {str(i+icon_start)}: {txt}")
    parsed_content_merged = ocr_text + parsed_content_icon_ls

    filtered_boxes = box_convert(boxes=filtered_boxes, in_fmt="xyxy", out_fmt="cxcywh")
    phrases = [i for i in range(len(filtered_boxes))]

    
    box_overlay_ratio = width / 3200
   
    
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }
    # draw boxes
    if draw_bbox_config:
        annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=filtered_boxes, logits=logits, phrases=phrases, **draw_bbox_config)
    else:
        annotated_frame, label_coordinates = annotate(image_source=image_source, boxes=filtered_boxes, logits=logits, phrases=phrases, text_scale=text_scale, text_padding=text_padding)

    
    pil_img = Image.fromarray(annotated_frame)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    pil_img.save("Annotated.png")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('ascii')
    if output_coord_in_ratio:
        label_coordinates = {k: [v[0]/w, v[1]/h, v[2]/w, v[3]/h] for k, v in label_coordinates.items()}
        assert w == annotated_frame.shape[1] and h == annotated_frame.shape[0]

    return encoded_image, label_coordinates , filtered_boxes_elem


def get_captions(croped_pil_images, resnet_model, florence_model, florence_processor,
                  faiss_path="faiss_index.bin", caption_cache_path="caption_cache.pkl",
                  similarity_threshold=0.8, batch_size=4):

    def load_or_initialize_caches(faiss_path, caption_cache_path, embedding_dim=None):
        """Load or initialize FAISS index and caption cache"""
        if os.path.exists(faiss_path) and os.path.exists(caption_cache_path):
            index = faiss.read_index(faiss_path)
            with open(caption_cache_path, "rb") as f:
                caption_cache = pickle.load(f)
        else:
            if embedding_dim is None:
                raise ValueError("Must provide embedding_dim when initializing new caches")
            index = faiss.IndexFlatIP(embedding_dim)
            caption_cache = {}
        return index, caption_cache
    
    def extract_embedding(resnet50, images_tensor):
        with torch.no_grad():
            embedding = resnet50(images_tensor).squeeze(-1).squeeze(-1)
        return embedding.numpy()

    def find_cached_captions(batch_embeddings, index, caption_cache, similarity_threshold=0.8):
        """Search cache for similar embeddings and return cached captions"""
        cached_indices = []
        cached_captions = []
        new_image_indices = list(range(len(batch_embeddings)))
        
        if index.ntotal > 0:
            # Search with cosine similarity
            distances, indices = index.search(batch_embeddings.astype('float32'), 1)
            
            new_image_indices = []
            for i, (distance, idx) in enumerate(zip(distances, indices)):
                if distance[0] >= similarity_threshold:
                    cached_indices.append(i)
                    cached_captions.append(caption_cache[int(idx[0])])
                else:
                    new_image_indices.append(i)
        
        return cached_indices, cached_captions, new_image_indices
        
    def save_to_caches(new_embeddings, new_captions, new_image_indices, 
                      index, caption_cache, faiss_path, caption_cache_path):
        """Update and persist caches with new embeddings and captions"""
        if not new_image_indices:
            return  # Nothing to update
        
        # Add new embeddings to FAISS index
        index.add(new_embeddings.astype('float32'))
        
        # Update caption cache with new entries
        start_idx = index.ntotal - len(new_image_indices)
        for i, caption in enumerate(new_captions):
            caption_cache[start_idx + i] = caption
        
        # Persist updated caches
        faiss.write_index(index, faiss_path)
        with open(caption_cache_path, "wb") as f:
            pickle.dump(caption_cache, f)

    
            
    # Convert PIL images to tensors
    transform = transforms.ToTensor()
    tensor_images = [transform(img) for img in croped_pil_images]
    batch_tensor = torch.stack(tensor_images)

    # Extract and normalize embeddings
    batch_embeddings = extract_embedding(resnet_model, batch_tensor)
    batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)

    # Load caches
    index, caption_cache = load_or_initialize_caches(
        faiss_path, caption_cache_path, embedding_dim=batch_embeddings.shape[1]
    )

    # Find cached captions
    cached_indices, cached_captions, new_image_indices = find_cached_captions(
        batch_embeddings, index, caption_cache, similarity_threshold
    )

    print(f"Uncached Icons = {len(new_image_indices)}")
    # Generate captions for new images
    new_captions = []
    if new_image_indices:
        new_images = [croped_pil_images[i] for i in new_image_indices]
        new_captions = get_icons_caption(
            florence_model, florence_processor, new_images, batch_size
        )
    
    # Merge results maintaining original order
    final_captions = [None] * len(croped_pil_images)
    for i, caption in zip(cached_indices, cached_captions):
        final_captions[i] = caption
    for i, caption in zip(new_image_indices, new_captions):
        final_captions[i] = caption

    # Save new entries to caches
    save_to_caches(
        batch_embeddings[new_image_indices], new_captions, new_image_indices,
        index, caption_cache, faiss_path, caption_cache_path
    )

    return final_captions


def get_parsed_icons_captions(
    screenshot_path,
    florence_model,
    florence_processor,
    yolo_model, 
    paddle_ocr,
    easyocr_reader,
    resnet_model 
   ):

    batch_size = 2
    
    # OCR Text and Boundary Box processing using easy oVcr and paddle ocr
    ocr_bbox_rslt, _ = check_ocr_box(screenshot_path, display_img=False, output_bb_format='xyxy', 
                                     goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
                                     use_paddleocr=True,paddle_ocr=paddle_ocr,reader=easyocr_reader)
    text, ocr_bbox = ocr_bbox_rslt

    # Icons Boundary Box Detection using YOLO
    xyxy, logits, phrases = yolo_icon_box_detection(screenshot_path, yolo_model)

    # Merge Overlappign YOLO Boxes and OCR Boxes
    starting_idx , filtered_boxes, filtered_boxes_elem = merge_ocr_yolo_boxes(screenshot_path,ocr_bbox, text, xyxy, iou_threshold=0.1)

    # Get croped images of detected icons from screenshot
    croped_pil_images = get_cropped_icons_images(starting_idx, filtered_boxes, screenshot_path)
    
    image_captions = get_captions(croped_pil_images, resnet_model, florence_model, florence_processor,batch_size=batch_size)

    # Draw bounding boxes around icons and annotate 
    encoded_image, label_coordinates, parsed_content_list = draw_annotate_boxes(screenshot_path, filtered_boxes_elem,
                                                                                filtered_boxes, text, image_captions,logits,phrases)

    # Extract Icons Coordinates
    icon_coordinates = [item['bbox'] for item in parsed_content_list]

    # Extract Icons Descriptions with an ID 
    icon_descriptions = '\n'.join([f'icon{i}:{str(v['content'])} ' for i,v in enumerate(parsed_content_list)])

    return encoded_image, icon_coordinates , icon_descriptions
    

def load_resnet_model(resnet_path):
    """Loads and optimizes a pre-trained ResNet model for CPU inference."""
    # Load ResNet model (ResNet-50 is a good balance of speed and accuracy)
    # Create a new instance of ResNet50
    model = resnet50()
    
    # Load the saved weights
    model.load_state_dict(torch.load(resnet_path))

    
    # Remove fully connected (FC) layer for feature extraction
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Removes the final classification layer
    
    # model = torch.compile(model)
    # Set model to evaluation mode
    model.eval()

    # Use TorchScript to optimize for CPU
    scripted_model = torch.jit.script(model)
    
    return scripted_model

def load_yolo_model(yolo_path):
    from ultralytics import YOLO
    # Load the model.
    model = YOLO(yolo_path)
    return model

def load_florence_model(florence_path):
    from transformers import AutoProcessor, AutoModelForCausalLM 
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(florence_path, torch_dtype=torch.float32, trust_remote_code=True)
    model = torch.compile(model)
    return model, processor

def load_ocr():
    reader = easyocr.Reader(['en'])
    paddle_ocr = PaddleOCR(
        lang='en',  # other lang also available
        use_angle_cls=False,
        use_gpu=False,  # using cuda will conflict with pytorch in the same process
        show_log=False,
        max_batch_size=1024,
        use_dilation=True,  # improves accuracy
        det_db_score_mode='slow',  # improves accuracy
        rec_batch_num=1024)

    return reader , paddle_ocr

