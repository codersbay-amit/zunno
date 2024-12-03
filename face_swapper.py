from face_predict import Predictor

predicter=Predictor()
predicter.setup()


def swap(source,target):
   source.save('source.jpg')
   target.save('target.jpg')
   output=predicter.predict('source.jpg','target.jpg')
   return output    

