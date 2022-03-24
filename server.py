import base64

from flask import Flask, request  # 서버 구현을 위한 Flask 객체 import
from flask_restx import Api, Resource  # Api 구현을 위한 Api 객체 import
import torch, torchvision  # torch 모듈 import
from torchvision import transforms
import mmdet  # mmdetection 모듈 import
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from werkzeug.datastructures import FileStorage
from PIL import Image

"""
    Swin transformer model initializer
"""
# Choose to use a config and initialize the detector
config = 'configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py'
# Setup a checkpoint file to load
checkpoint = 'checkpoints/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth'
# initialize the detector
model = init_detector(config, checkpoint, device='cpu:0')
"""
    coatnet model initializer
"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# load a pre=trained model for coatnet
model = torch.load('./savemodel/dogbreed2.pth')
model.to(device)
model.eval()

app = Flask(__name__)  # Flask 객체 선언, 파라미터로 어플리케이션 패키지의 이름을 넣어줌.
api = Api(app)  # Flask 객체에 Api 객체 등록

upload_parser = api.parser()
upload_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)


@api.route('/predict')  # 데코레이터 이용, '/predict' 경로에 클래스 등록
class HelloWorld(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_name = 0

    def post(self):
        # 이미지 전송 받기
        args = upload_parser.parse_args()
        # 이미지 전송 받은 파일을 이미지 파일로 변환
        uploaded_file = args['file']
        uploaded_file.save(f'imgs/find{self.file_name}.jpg')
        result = inference_detector(model, './imgs/find' + str(self.file_name) + '.jpg')
        find = any([any(result[0][15].T[4] > 0.3), any(result[0][16].T[4] > 0.3)])  # find cats and dogs
        self.file_name = (self.file_name + 1) % 20
        # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        return {"predict": find}


@api.route('/predict/breed/dog')  # 데코레이터 이용, '/predict' 경로에 클래스 등록
class HelloWorld(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_name = 0
        self.transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.classes = ["Chihuaha",
                        "Japanese Spaniel",
                        "Maltese Dog",
                        "Pekinese",
                        "Shih-Tzu",
                        "Blenheim Spaniel",
                        "Papillon",
                        "Toy Terrier",
                        "Rhodesian Ridgeback",
                        "Afghan Hound",
                        "Basset Hound",
                        "Beagle",
                        "Bloodhound",
                        "Bluetick",
                        "Black-and-tan Coonhound",
                        "Walker Hound",
                        "English Foxhound",
                        "Redbone",
                        "Borzoi",
                        "Irish Wolfhound",
                        "Italian Greyhound",
                        "Whippet",
                        "Ibizian Hound",
                        "Norwegian Elkhound",
                        "Otterhound",
                        "Saluki",
                        "Scottish Deerhound",
                        "Weimaraner",
                        "Staffordshire Bullterrier",
                        "American Staffordshire Terrier",
                        "Bedlington Terrier",
                        "Border Terrier",
                        "Kerry Blue Terrier",
                        "Irish Terrier",
                        "Norfolk Terrier",
                        "Norwich Terrier",
                        "Yorkshire Terrier",
                        "Wirehaired Fox Terrier",
                        "Lakeland Terrier",
                        "Sealyham Terrier",
                        "Airedale",
                        "Cairn",
                        "Australian Terrier",
                        "Dandi Dinmont",
                        "Boston Bull",
                        "Miniature Schnauzer",
                        "Giant Schnauzer",
                        "Standard Schnauzer",
                        "Scotch Terrier",
                        "Tibetan Terrier",
                        "Silky Terrier",
                        "Soft-coated Wheaten Terrier",
                        "West Highland White Terrier",
                        "Lhasa",
                        "Flat-coated Retriever",
                        "Curly-coater Retriever",
                        "Golden Retriever",
                        "Labrador Retriever",
                        "Chesapeake Bay Retriever",
                        "German Short-haired Pointer",
                        "Vizsla",
                        "English Setter",
                        "Irish Setter",
                        "Gordon Setter",
                        "Brittany",
                        "Clumber",
                        "English Springer Spaniel",
                        "Welsh Springer Spaniel",
                        "Cocker Spaniel",
                        "Sussex Spaniel",
                        "Irish Water Spaniel",
                        "Kuvasz",
                        "Schipperke",
                        "Groenendael",
                        "Malinois",
                        "Briard",
                        "Kelpie",
                        "Komondor",
                        "Old English Sheepdog",
                        "Shetland Sheepdog",
                        "Collie",
                        "Border Collie",
                        "Bouvier des Flandres",
                        "Rottweiler",
                        "German Shepard",
                        "Doberman",
                        "Miniature Pinscher",
                        "Greater Swiss Mountain Dog",
                        "Bernese Mountain Dog",
                        "Appenzeller",
                        "EntleBucher",
                        "Boxer",
                        "Bull Mastiff",
                        "Tibetan Mastiff",
                        "French Bulldog",
                        "Great Dane",
                        "Saint Bernard",
                        "Eskimo Dog",
                        "Malamute",
                        "Siberian Husky",
                        "Affenpinscher",
                        "Basenji",
                        "Pug",
                        "Leonberg",
                        "Newfoundland",
                        "Great Pyrenees",
                        "Samoyed",
                        "Pomeranian",
                        "Chow",
                        "Keeshond",
                        "Brabancon Griffon",
                        "Pembroke",
                        "Cardigan",
                        "Toy Poodle",
                        "Miniature Poodle",
                        "Standard Poodle",
                        "Mexican Hairless",
                        "Dingo",
                        "Dhole",
                        "African Hunting Dog"]

    def post(self):
        # 이미지 전송 받기
        args = upload_parser.parse_args()
        # 이미지 전송 받은 파일을 이미지 파일로 변환
        uploaded_file = args['file']
        path = f'imgs/dog{self.file_name}.jpg'
        uploaded_file.save(path)
        img = Image.open(path)
        output = model(img.unsqueeze(0).to(device))
        top3 = torch.topk(output, 3, dim=1)
        predict_list = [int(x) for x in top3.indices.squeeze()]  # find dog breed
        self.file_name = (self.file_name + 1) % 20
        # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        return {"predicts": predict_list}


@api.route('/hello')  # 데코레이터 이용, '/predict' 경로에 클래스 등록
class HelloWorld(Resource):
    def get(self):
        # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        return {"hello": "world!"}


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8081)
