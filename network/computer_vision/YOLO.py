import torch
import torch.nn as nn

class YOLOBackbone(nn.Module):
    def __init__(self):
        super(YOLOBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = self.pool(nn.ReLU()(self.conv4(x)))
        x = self.pool(nn.ReLU()(self.conv5(x)))
        return x

class YOLODetectionHead(nn.Module):
    def __init__(self, grid_size=8, num_classes=2):
        super(YOLODetectionHead, self).__init__()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Conv2d(128, 5 + num_classes, kernel_size=1)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = self.fc(x)  # Output shape: (batch, grid_size, grid_size, 5+num_classes)
        return x.view(x.size(0), 8, 8, 5 + 2)  # Reshape to (batch, grid_size, grid_size, 5+num_classes)

class YoloLoss(nn.Module):
    def __init__(self, lambda_coord=5, lambda_noobj=0.5):
        super(YoloLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        # Extract values
        pred_boxes = predictions[..., :4]  # (x, y, w, h)
        pred_conf = predictions[..., 4]  # Object confidence score
        pred_class = predictions[..., 5:]  # Class probabilities

        target_boxes = targets[..., :4]
        target_conf = targets[..., 4]
        target_class = targets[..., 5:]

        # Bounding box loss
        box_loss = self.lambda_coord * self.mse(pred_boxes, target_boxes)

        # Objectness loss
        obj_loss = self.bce(pred_conf, target_conf)

        # Classification loss
        class_loss = self.ce(pred_class, target_class)

        total_loss = box_loss + obj_loss + class_loss
        return total_loss



if __name__ == '__main__':
    # Sample input
    input_tensor = torch.rand(1, 3, 256, 256)  # Batch size = 1, Image size = 256x256
    model = YOLOBackbone()
    feature_map = model(input_tensor)
    print("Feature Map Shape:", feature_map.shape)

    detection_head = YOLODetectionHead(grid_size=8, num_classes=2)
    predictions = detection_head(feature_map)
    print("YOLO Output Shape:", predictions.shape)

    # Sample usage
    loss_fn = YoloLoss()
    sample_preds = torch.rand(1, 8, 8, 7)  # Dummy predictions
    sample_targets = torch.rand(1, 8, 8, 7)  # Dummy ground truth
    loss = loss_fn(sample_preds, sample_targets)
    print("Sample Loss:", loss.item())
