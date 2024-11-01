from PIL import ImageDraw, ImageFont
import random
import matplotlib.pyplot as plt

def load_model(model_path, model_class, device):
    model = model_class(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def get_prediction(model, image_tensor, device):
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        _, predicted = outputs.max(1)
    return predicted.item()

def visualize_prediction(image_path, roi_data, bbox_data, model, transform, device, actual_label):
    # Load and preprocess the image
    original_image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess_image(image_path, transform)
    
    # Get model prediction
    predicted_label = get_prediction(model, image_tensor, device)
    
    # Get ROI and bbox information
    img_name = image_path.split('/')[-1]
    roi = roi_data.get(img_name, {}).get('Boxes', [None])[0]
    bbox = bbox_data.get(img_name, {}).get('Boxes', [None])[0]
    
    # Create a copy of the image for drawing
    draw_image = original_image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # Draw ROI box in blue
    if roi:
        draw.rectangle(roi, outline="blue", width=3)
        draw.text((roi[0], roi[1] - 20), "ROI", fill="blue")
    
    # Draw ground truth bbox in green
    if bbox:
        draw.rectangle(bbox, outline="green", width=3)
        draw.text((bbox[0], bbox[1] - 20), "Ground Truth", fill="green")
    
    # Add prediction and actual label text
    draw.text((10, 10), f"Predicted: {predicted_label}", fill="red")
    draw.text((10, 30), f"Actual: {actual_label}", fill="green")
    
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(draw_image)
    plt.axis('off')
    plt.title(f"Prediction: {predicted_label}, Actual: {actual_label}")
    plt.show()

def visualize_random_samples(num_samples, model, test_dataset, roi_data, bbox_data, device):
    indices = random.sample(range(len(test_dataset)), num_samples)
    
    for idx in indices:
        image_path, label = test_dataset.data[idx]
        full_image_path = f"{test_dataset.img_dir}/{image_path}"
        
        visualize_prediction(
            full_image_path,
            roi_data,
            bbox_data,
            model,
            test_dataset.transform,
            device,
            label
        )

# Usage example
def main_show_vgg16():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model (you can choose between VGG16 and VGG19)
    model = load_model('vgg16_best_model.pth', ModifiedVGG16, device)
    
    # Prepare the test dataset
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = GBCUDataset('/kaggle/input/gbcu-data/GBCU-Shared/test.txt', 
                               '/kaggle/input/gbcu-data/GBCU-Shared/imgs', 
                               r"/kaggle/input/gbcu-data/GBCU-Shared/roi_pred.json", 
                               r"/kaggle/input/gbcu-data/GBCU-Shared/bbox_annot.json", 
                               transform=test_transform)
    
    # Load ROI and bbox data
    with open(r"/kaggle/input/gbcu-data/GBCU-Shared/roi_pred.json", 'r') as f:
        roi_data = json.load(f)
    with open(r"/kaggle/input/gbcu-data/GBCU-Shared/bbox_annot.json", 'r') as f:
        bbox_data = json.load(f)
    
    # Visualize predictions for 5 random samples
    visualize_random_samples(5, model, test_dataset, roi_data, bbox_data, device)

def main_show_vgg19():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model (you can choose between VGG16 and VGG19)
    model = load_model('vgg19_best_model.pth', ModifiedVGG19, device)
    
    # Prepare the test dataset
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = GBCUDataset('/kaggle/input/gbcu-data/GBCU-Shared/test.txt', 
                               '/kaggle/input/gbcu-data/GBCU-Shared/imgs', 
                               r"/kaggle/input/gbcu-data/GBCU-Shared/roi_pred.json", 
                               r"/kaggle/input/gbcu-data/GBCU-Shared/bbox_annot.json", 
                               transform=test_transform)
    
    # Load ROI and bbox data
    with open(r"/kaggle/input/gbcu-data/GBCU-Shared/roi_pred.json", 'r') as f:
        roi_data = json.load(f)
    with open(r"/kaggle/input/gbcu-data/GBCU-Shared/bbox_annot.json", 'r') as f:
        bbox_data = json.load(f)
    
    # Visualize predictions for 5 random samples
    visualize_random_samples(5, model, test_dataset, roi_data, bbox_data, device)

main_show_vgg16()
main_show_vgg19()
