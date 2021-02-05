#include<opencv2/opencv.hpp>
#include<opencv2/face.hpp>
#include<fstream>
#include <iostream>
#include<vector>

using namespace cv;
using namespace std;
using namespace cv::face;

void drawText(Mat & image);

int main()
{
	Mat image,testSample, dst;
	CascadeClassifier faceDetector;
	faceDetector.load("haarcascade_frontalface_alt2.xml");
	string s, text;int count = 0, label;
	vector<Mat> images;
	vector<int> labels;
	ifstream ifs("rawData");
	while(ifs>>s){
		images.push_back(imread(s.c_str(), 0));
		ifs>>s;
		labels.push_back(atoi(s.c_str()));
	}
	ifs.close();

	
	Ptr<BasicFaceRecognizer> model = EigenFaceRecognizer::create();
	model->train(images, labels);
	
	VideoCapture capture(0);
	if(!capture.isOpened()){
		cout<<"ERROR: Can Not Open The Default Camera, Please Check Your Device!";
		return -1;
	}
	
	vector<Rect> faces;
	while(capture.read(image)){
		faceDetector.detectMultiScale(image, faces, 1.1, 3, 0, Size(30,30));
		for(int i = 0;i < faces.size(); i++){
			rectangle(image, Point(faces[i].x, faces[i].y), Point(faces[i].x+faces[i].width, faces[i].y+faces[i].height), Scalar(0, 0, 255));

			cvtColor(image(faces[i]), dst, COLOR_BGR2GRAY);

			resize(dst, testSample, images[0].size());

			label = model->predict(testSample);

			if(label == 1)
				text = "Obama";
			else if(label == 2)
				text = "Trump";
			else if(label == 3)
				text = "JinPing Xi";
			else if(label == 4)
				text = "Yun Dong";
			else
				text = "Stranger";
			putText(image, text, Point(faces[i].x, faces[i].y-10), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0));
		}
		imshow("test", image);
		if(waitKey(7)==27){
			break;
		}
	}
    capture.release();
    
    return 0;
}
