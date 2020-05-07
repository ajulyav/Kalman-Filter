/*
 * FgSegment.cpp
 */

#include "FgSegment.hpp"

// Parameters for MOG for streetCornerAtNight_0_100_clip
#define HISTORY 50
#define VARTHERSHOLD 16
#define LEARNING_RATE 0.001



// Parameters of Opening
/* Kernel Size
 * Not necessarily important for most sequences, except streetCornerAtNight_0_100_clip where light changes
 * provoke a fake identification of blobs on the lower part of the screen. Here Kernel should be 5
 * --> The higher the kernel the more noise we can get rid off. You can also get smaller blobs that later
 * --> on will be deleted by the delete small blobs function
 * #define KERNEL 5
 */

#define KERNEL 3
int operation = MORPH_OPEN;
int type = MORPH_RECT;

int connectivity = 4;


Ptr<BackgroundSubtractor> pMOG2 = cv::createBackgroundSubtractorMOG2(HISTORY, VARTHERSHOLD, true);

// background subtraction
Mat FgSegment::bkg_subtraction(Mat frame){

	//background subtraction

	Mat fgmask = Mat::zeros(frame.size(),frame.type());
	pMOG2->apply(frame, fgmask, LEARNING_RATE);

	return fgmask;
}


void FgSegment::extractBlobs(Mat fgmask)
{

	 Mat aux; // image to be updated each time a blob is detected (blob cleared)
	 fgmask.convertTo(aux, CV_32SC1);
     bloblist.clear(); //clear blob list

     int area, new_area = 0;
     int index, x, new_x, y, new_y, width, new_width, height,new_height = 0;

	 double center_x, center_y, new_center_x, new_center_y = 0;


	 Mat stats , centroids;
	 int new_blob =  connectedComponentsWithStats(fgmask, aux, stats, centroids, connectivity);

	 for (int i = 0 ; i < new_blob ; i++)

	 {
		 //height and width
		  width = stats.at<int>(i, CC_STAT_WIDTH);
		  height = stats.at<int>(i, CC_STAT_HEIGHT);

		 // check sizes
		 if(height > MIN_HEIGHT && width > MIN_WIDTH)
		 {
			 if(height != fgmask.rows && width != fgmask.cols)
			 {
				 x = stats.at<int>(i, CC_STAT_LEFT);
				 y = stats.at<int>(i, CC_STAT_TOP);

				 center_x = centroids.at<double>(i,0);
				 center_y = centroids.at<double>(i,1);

				 area = stats.at<int>(i, CC_STAT_AREA);

				  if(new_area <= area)
				  {
		              new_x = x;
		              new_y = y;
					  new_center_x = center_x;
					  new_center_y = center_y;

		              new_height = height;
		              new_width = width;

					  new_area = area;
		              index = i;
				  }
			 }
		 }

         }

	 if(new_area != 0)
	 {

	   cvBlob blob = initBlob(index,new_x, new_y, new_width, new_height,new_center_x,new_center_y, new_area);

	   bloblist.push_back(blob);
       blob_centers.push_back(Point(new_center_x,new_center_y));

       blob_extracted = true;
	 }
}

//morphological opening
Mat FgSegment::MorphologicalOpen(Mat fgmask)

{
	Mat element = getStructuringElement(type, Size(KERNEL, KERNEL),Point(1,1));
	morphologyEx(fgmask, fgmask, operation, element);

	return fgmask;

}

Mat FgSegment::paintBlobImage(cv::Mat frame, std::vector<cvBlob> bloblist)
{
	cv::Mat blobImage;
		//check input conditions and return original if any is not satisfied
		//...
	frame.copyTo(blobImage);

	//paint each blob of the list
		for(int i = 0; i < bloblist.size(); i++)
		{
			cvBlob blob = bloblist[i]; //get ith blob


			Point p1 = Point(blob.x, blob.y);
			Point p2 = Point(blob.x+blob.w, blob.y+blob.h);

			rectangle(blobImage, p1, p2, Scalar(51, 255, 204), 1, 1, 0);

		}


	//return the image to show
	return blobImage;
}
