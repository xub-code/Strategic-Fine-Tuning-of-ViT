import os
import json
import torch
from PIL import Image
from torchvision import transforms

from vit_model import vit_base_patch16_224_in21k as create_model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device for testing.")

    num_classes = 3
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5],
                              [0.5, 0.5, 0.5])])

    test_dir = "val_test"
    assert os.path.exists(test_dir), f"Directory '{test_dir}' does not exist."

    # Read class_indices.json file
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "File '{}' does not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # Create the model
    model = create_model(num_classes=num_classes, has_logits=False).to(device)
    # Load model weights
    model_weight_path = "./weights/last.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    # Create a mapping dictionary from class_folder to class_label
    class_folder_to_label = {}
    for class_label, class_name in class_indict.items():
        class_folder_to_label[class_name] = int(class_label)

    # Test the model on the test dataset
    correct = 0
    total = 0

    # Initialize a dictionary to store correct predictions and total images for each class
    class_correct = {class_name: 0 for class_name in class_indict.values()}
    class_total = {class_name: 0 for class_name in class_indict.values()}

    for class_folder in os.listdir(test_dir):
        class_folder_path = os.path.join(test_dir, class_folder)
        if os.path.isdir(class_folder_path):
            true_class = class_folder_to_label.get(class_folder, None)
            if true_class is not None:
                for img_file in os.listdir(class_folder_path):
                    img_path = os.path.join(class_folder_path, img_file)
                    assert os.path.exists(img_path), f"File '{img_path}' does not exist."
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = data_transform(img)
                    img_tensor = torch.unsqueeze(img_tensor, dim=0)
                    img_tensor = img_tensor.to(device)

                    with torch.no_grad():
                        output = model(img_tensor)
                        predict = torch.softmax(output, dim=1)
                        predict_class = torch.argmax(predict, dim=1)

                    total += 1
                    class_total[class_folder] += 1  # Increment total images for this class
                    if predict_class.item() == true_class:
                        correct += 1
                        class_correct[class_folder] += 1  # Increment correct predictions for this class

    # Calculate and print accuracy for each class
    for class_name, correct_count in class_correct.items():
        total_count = class_total[class_name]
        class_accuracy = 100 * correct_count / total_count
        print(f"Accuracy for class '{class_name}': {class_accuracy:.3f}%")

    # Calculate and print overall accuracy
    accuracy = 100 * correct / total
    print(f"Overall accuracy on the test dataset: {accuracy:.3f}%")

if __name__ == '__main__':
    main()
