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
checkpoint = 'models/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth'
# initialize the detector
model = init_detector(config, checkpoint, device='cpu:0')
"""
    coatnet model initializer
"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# load a pre=trained model for coatnet
dog_model = torch.load('models/dogbreed2.pth', map_location=device)
dog_model.to(device)
dog_model.eval()
cat_model = torch.load('models/catbreed3.pth', map_location=device)
cat_model.to(device)
cat_model.eval()
emotion_model = torch.load('models/emotion1.pth', map_location=device)
emotion_model.to(device)
emotion_model.eval()

app = Flask(__name__)  # Flask 객체 선언, 파라미터로 어플리케이션 패키지의 이름을 넣어줌.
api = Api(app)  # Flask 객체에 Api 객체 등록

upload_parser = api.parser()
upload_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)

cat_file_name = 0
dog_file_name = 0
emotion_file_name = 0
detect_file_name = 0


@api.route('/detect', methods=['GET', 'POST'])
class FindCatsAndDogs(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_name = 0

    def post(self):
        # 이미지 전송 받기
        args = upload_parser.parse_args()
        # 이미지 전송 받은 파일을 이미지 파일로 변환
        uploaded_file = args['file']
        uploaded_file.save(f'imgs/find{self.file_name}.jpg')
        try:
            result = inference_detector(model, './imgs/find' + str(self.file_name) + '.jpg')
            find = "false"
            if all([any(result[0][15].T[4] > 0.3), any(result[0][16].T[4] > 0.3)]):  # 개와 고양이 모두가 있으면
                if (result[0][15].T[4][0] < result[0][16].T[4][0]):  # 고양이가 개보다 확률 up?
                    find = "cat"
                else:  # 개가 확률이 더 높다
                    find = "dog"
            else:
                if any(result[0][15].T[4] > 0.3):  # 고양이가 있으면
                    find = "cat"
                if any(result[0][16].T[4] > 0.3):  # 개가 있으면
                    find = "dog"

            return {"detected": find}
        except:
            json_object = {"message": "Image analysis error"}

        self.file_name = (self.file_name + 1) % 20
        # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        return json_object

    def get(self):
        path = request.args.get('path')
        if path is None:
            return {"error": "The parameter path is not exist."}
        try:
            result = inference_detector(model, path)
            find = "false"
            if all([any(result[0][15].T[4] > 0.3), any(result[0][16].T[4] > 0.3)]):  # 개와 고양이 모두가 있으면
                if (result[0][15].T[4][0] < result[0][16].T[4][0]):  # 고양이가 개보다 확률 up?
                    find = "cat"
                else:  # 개가 확률이 더 높다
                    find = "dog"
            else:
                if any(result[0][15].T[4] > 0.3):  # 고양이가 있으면
                    find = "cat"
                if any(result[0][16].T[4] > 0.3):  # 개가 있으면
                    find = "dog"

            return {"detected": find}
        except:
            json_object = {"message": "Image analysis error"}

        # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        return json_object


@api.route('/predict/breed/dog', methods=['GET', 'POST'])
class DistinguishDogBreed(Resource):
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
        global dog_file_name
        # 이미지 전송 받기
        args = upload_parser.parse_args()
        # 이미지 전송 받은 파일을 이미지 파일로 변환
        uploaded_file = args['file']
        path = f'imgs/dog{dog_file_name}.jpg'
        uploaded_file.save(path)
        try:
            img = Image.open(path).convert('RGB')
            result = inference_detector(model, path)
            json_object = {"error": "Can't find a dog"}
            if any(result[0][16][:, 4] > 0.3):
                position = [int(i) for i in result[0][16][0]]
                img = img.crop(position[:-1])
                img = self.transformer(img)
                output = dog_model(img.unsqueeze(0).to(device))
                top3 = torch.topk(output, 3, dim=1)
                predict_list = [self.classes[x] for x in top3.indices.squeeze()]  # find dog breed
                values_list = [float(x) for x in top3.values.squeeze()]
                json_object = {"crop_position": position[:-1],
                               "top3": [{"breed": predict_list[0], "value": values_list[0]},
                                        {"breed": predict_list[1], "value": values_list[1]},
                                        {"breed": predict_list[2], "value": values_list[2]}]}
        except:
            json_object = {"message": "Image analysis error"}
            dog_file_name = (dog_file_name + 1) % 20
        # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        return json_object

    def get(self):
        path = request.args.get('path')
        if path is None:
            return {"error": "The parameter path is not exist."}
        try:
            img = Image.open(path).convert('RGB')
            result = inference_detector(model, path)
            json_object = {"error": "Can't find a dog"}
            if any(result[0][16][:, 4] > 0.3):
                position = [int(i) for i in result[0][16][0]]
                img = img.crop(position[:-1])
                img = self.transformer(img)
                output = dog_model(img.unsqueeze(0).to(device))
                top3 = torch.topk(output, 3, dim=1)
                predict_list = [self.classes[x] for x in top3.indices.squeeze()]  # find dog breed
                values_list = [float(x) for x in top3.values.squeeze()]
                json_object = {"crop_position": position[:-1],
                               "top3": [{"breed": predict_list[0], "value": values_list[0]},
                                        {"breed": predict_list[1], "value": values_list[1]},
                                        {"breed": predict_list[2], "value": values_list[2]}]}
        except:
            json_object = {"message": "Image analysis error"}

        # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        return json_object


@api.route('/predict/breed/cat')  # 데코레이터 이용, '/predict' 경로에 클래스 등록
class DistinguishCatBreed(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_name = 0
        self.transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.classes = ['bengal', 'british shorthair', 'domestic long-haired', 'domestic short-haired', 'maine coon',
                        'Munchkin', 'Norwegian forest', 'persian', 'ragdoll', 'russian blue', 'scottish fold',
                        'selkirk rex', 'siamese', 'sphynx']

    def post(self):
        global cat_file_name
        # 이미지 전송 받기
        args = upload_parser.parse_args()
        # 이미지 전송 받은 파일을 이미지 파일로 변환
        uploaded_file = args['file']
        path = f'imgs/cat{cat_file_name}.jpg'
        uploaded_file.save(path)
        try:
            img = Image.open(path).convert('RGB')
            result = inference_detector(model, path)
            json_object = {"error": "Can't find a cat"}
            if any(result[0][15][:, 4] > 0.3):
                position = [int(i) for i in result[0][15][0]]
                img = img.crop(position[:-1])
                img = self.transformer(img)
                output = cat_model(img.unsqueeze(0).to(device))
                top3 = torch.topk(output, 3, dim=1)
                predict_list = [self.classes[x] for x in top3.indices.squeeze()]  # find cat breed
                values_list = [float(x) for x in top3.values.squeeze()]
                json_object = {"crop_position": position[:-1],
                               "top3": [{"breed": predict_list[0], "value": values_list[0]},
                                        {"breed": predict_list[1], "value": values_list[1]},
                                        {"breed": predict_list[2], "value": values_list[2]}]}
        except:
            json_object = {"message": "Image analysis error"}
            cat_file_name = (cat_file_name + 1) % 20
        # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        return json_object

    def get(self):
        path = request.args.get('path')
        if path is None:
            return {"error": "The parameter path is not exist."}
        try:
            img = Image.open(path).convert('RGB')
            result = inference_detector(model, path)
            json_object = {"error": "Can't find a cat"}
            if any(result[0][15][:, 4] > 0.3):
                position = [int(i) for i in result[0][15][0]]
                img = img.crop(position[:-1])
                img = self.transformer(img)
                output = cat_model(img.unsqueeze(0).to(device))
                top3 = torch.topk(output, 3, dim=1)
                predict_list = [self.classes[x] for x in top3.indices.squeeze()]  # find cat breed
                values_list = [float(x) for x in top3.values.squeeze()]
                json_object = {"crop_position": position[:-1],
                               "top3": [{"breed": predict_list[0], "value": values_list[0]},
                                        {"breed": predict_list[1], "value": values_list[1]},
                                        {"breed": predict_list[2], "value": values_list[2]}]}
        except:
            json_object = {"message": "Image analysis error"}
        # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        return json_object


@api.route('/predict/emotion', methods=['GET', 'POST'])  # 데코레이터 이용, '/predict' 경로에 클래스 등록
class GuessEmotion(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.classes = ['angry', 'happy', 'sad']

    def post(self):
        global emotion_file_name
        # 이미지 전송 받기
        args = upload_parser.parse_args()
        # 이미지 전송 받은 파일을 이미지 파일로 변환
        uploaded_file = args['file']
        path = f'imgs/emotion{emotion_file_name}.jpg'
        uploaded_file.save(path)
        try:
            img = Image.open(path).convert('L')
            result = inference_detector(model, path)
            json_object = {"message": "Can't find cat or dog"}
            if any(result[0][15][:, 4] > 0.3):
                position = [int(i) for i in result[0][15][0]]
                img = img.crop(position[:-1])
                img = self.transformer(img)
                output = emotion_model(img.unsqueeze(0).to(device))
                outputs = [float(x) for x in output[0]]
                json_object = {"message": "success!",
                               "category": "cat", "crop_position": position[:-1],
                               "emotion": {"angry": outputs[0], "happy": outputs[1], "sad": outputs[2]}}
                return json_object
            if any(result[0][16][:, 4] > 0.3):
                position = [int(i) for i in result[0][16][0]]
                img = img.crop(position[:-1])
                img = self.transformer(img)
                output = emotion_model(img.unsqueeze(0).to(device))
                outputs = [float(x) for x in output[0]]
                json_object = {"message": "success!",
                               "category": "dog", "crop_position": position[:-1],
                               "emotion": {"angry": outputs[0], "happy": outputs[1], "sad": outputs[2]}}
                return json_object
        except:
            json_object = {"message": "Image analysis error"}

            emotion_file_name = (emotion_file_name + 1) % 20
        # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        return json_object

    def get(self):
        path = request.args.get('path')
        if path is None:
            return {"message": "The parameter path is not exist."}
        try:
            img = Image.open(path).convert('L')
            result = inference_detector(model, path)
            json_object = {"message": "Can't find cat or dog"}
            if any(result[0][15][:, 4] > 0.3):
                position = [int(i) for i in result[0][15][0]]
                img = img.crop(position[:-1])
                img = self.transformer(img)
                output = emotion_model(img.unsqueeze(0).to(device))
                outputs = [float(x) for x in output[0]]
                json_object = {"message": "success!",
                               "category": "cat", "crop_position": position[:-1],
                               "emotion": {"angry": outputs[0], "happy": outputs[1], "sad": outputs[2]}}
                return json_object
            if any(result[0][16][:, 4] > 0.3):
                position = [int(i) for i in result[0][16][0]]
                img = img.crop(position[:-1])
                img = self.transformer(img)
                output = emotion_model(img.unsqueeze(0).to(device))
                outputs = [float(x) for x in output[0]]
                json_object = {"message": "success!",
                               "category": "dog", "crop_position": position[:-1],
                               "emotion": {"angry": outputs[0], "happy": outputs[1], "sad": outputs[2]}}
                return json_object
        except:
            json_object = {"message": "Image analysis error"}

        # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        return json_object


@api.route('/hello')  # 데코레이터 이용, '/predict' 경로에 클래스 등록
class HelloWorld(Resource):
    def get(self):
        # GET 요청시 리턴 값에 해당 하는 dict를 JSON 형태로 반환
        return {"hello": "world!"}


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=34343)
