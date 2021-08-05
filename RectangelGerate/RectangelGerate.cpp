// RectangelGerate.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include <random>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ctime>

int random(int upLim = 100, int downLim = 0)
{
    static int i{ 0 };
    if (i == 0)
    {
        srand(time(0));
        ++i;
    }
    if ((upLim - downLim) == 0)
        return downLim;
    else
        return 1 + rand() % (upLim - downLim) + downLim;
}

void findHeigtAndWeigth(cv::Point const *vertices, int const numberPoints, float &heigt, float &width)
{
    int minX{ vertices->x };
    int maxX{ minX };
    int minY{ vertices->y };
    int maxY{ minY };
    for (size_t i{ 1 }; i < numberPoints; ++i)
    {
        if (minX > (vertices + i)->x)
            minX = (vertices + i)->x;
        if (maxX < (vertices + i)->x)
            maxX = (vertices + i)->x;
        if (minY > (vertices + i)->y)
            minY = (vertices + i)->y;
        if (maxY < (vertices + i)->y)
            maxY = (vertices + i)->y;
    }
    heigt = maxY - minY;
    width = maxX - minX;
}

void bacgroundGenerate(cv::Mat &inOutput,int *outObjectColor, double const NSR, double SKO, int const *medium, bool *isPositivCantrast, bool noiseON =true)
{ 
    if (SKO <= 0)
    {
        SKO = 1;
        noiseON = false;
    }
    int chanels{ inOutput.channels() };

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::vector<std::normal_distribution<float>> dist{};
    if (chanels > 1)
    {
        dist.push_back(std::normal_distribution<float>(*medium, SKO));
        dist.push_back(std::normal_distribution<float>(*(medium + 1), SKO));
        dist.push_back(std::normal_distribution<float>(*(medium + 2), SKO));
    }
    else
    {
        dist.push_back(std::normal_distribution<float>(*medium, SKO));
    }

    int overflow{ 0 };
    for (size_t i{ 0 }; i < chanels; ++i)
    {
        if (*(medium+i) * NSR >= 255)
        {
            overflow = 1;
        }
        if (*(medium+i) / NSR <= 0)
        {
            overflow += 10;
        }
        if (overflow == 0)
        {
            if (isPositivCantrast)
            {
                *(outObjectColor + i) = *(medium + i) * NSR;
            }
            else
            {
                *(outObjectColor + i) = *(medium + i) / NSR;
            }
        }
        else if (overflow == 1)
        {
            *(outObjectColor + i) = *(medium + i) / NSR;
        }
        else if (overflow == 10)
        {
            *(outObjectColor + i) = *(medium + i) * NSR;
        }
        else
        {
            if (isPositivCantrast)
            {
                *(outObjectColor + i) = 255;
            }
            else
            {
                *(outObjectColor + i) = 0;
            }
        }
    }

    for (size_t x{ 0 }; x < chanels; ++x)
    {
        for (size_t i{ 0 }; i < inOutput.rows; ++i)
        {
            for (size_t j{ 0 }; j < inOutput.cols; ++j)
            {
                if (noiseON)
                {
                    float val = dist[x](gen);
                    if (val > 255.0)
                    {
                        val = 255.0;
                    }
                    else if (val < 0.0)
                    {
                        val = 0.0;
                    }
                    if (inOutput.channels() == 1)
                    {
                        inOutput.at<uchar>(i, j) = static_cast<unsigned char>(val);
                    }
                    else
                    {
                        inOutput.at<cv::Vec3b>(i, j)[x] = static_cast<unsigned char>(val);       
                    }
                }
                else
                {
                    if (inOutput.channels() == 1)
                    {
                        inOutput.at<uchar>(i, j) = static_cast<unsigned char>(*(medium+i));
                    }
                    else
                    {
                        inOutput.at<cv::Vec3b>(i, j)[x] = static_cast<unsigned char>(*(medium+i));
                    }
                }
            }
        }
    }

}

cv::Mat objectGenerate(cv::Mat input, double SKO, int const *medium, bool noiseON = true)
{
    if (SKO <= 0)
    {
        SKO = 1;
        noiseON = false;
    }
    int chanels{ input.channels() };

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::vector<std::normal_distribution<float>> dist{};
    if (chanels > 1)
    {
        dist.push_back(std::normal_distribution<float>(*medium, SKO));
        dist.push_back(std::normal_distribution<float>(*(medium + 1), SKO));
        dist.push_back(std::normal_distribution<float>(*(medium + 2), SKO));
    }
    else
    {
        dist.push_back(std::normal_distribution<float>(*medium, SKO));
    }

    cv::Mat buferOjectMat(input.size(), input.type());
    for (size_t x{ 0 }; x < chanels; ++x)
    {
        for (int i{ 0 }; i < input.rows; ++i)
        {
            for (size_t j{}; j < input.cols; ++j)
            {
                if (noiseON)
                {
                    float val = dist[x](gen);
                    if (val > 255.0)
                    {
                        val = 255.0;
                    }
                    else if (val < 0.0)
                    {
                        val = 0.0;
                    }
                    if (buferOjectMat.channels() == 1)
                    {
                        buferOjectMat.at<uchar>(i, j) = static_cast<unsigned char>(val);
                    }
                    else
                    {
                        buferOjectMat.at<cv::Vec3b>(i, j)[x] = static_cast<unsigned char>(val);
                    }
                }
                else
                {
                    if (buferOjectMat.channels() == 1)
                    {
                        buferOjectMat.at<uchar>(i, j) = static_cast<unsigned char>(*(medium+i));
                    }
                    else
                    {
                        buferOjectMat.at<cv::Vec3b>(i, j)[x] = static_cast<unsigned char>(*(medium+1));
                    }
                }
            }
        }
    }
    cv::bitwise_and(input, buferOjectMat, buferOjectMat);
    return buferOjectMat;
}

int main()
{
    double const pi{ 3.14159 };
    int imageSize{ 416 };
    int N{ 10 };
    int figureType{ 1 }; /// 0 - not figure 1 - square, 2 - circle, 3-triangel 
    bool isColorImg{ false };
    int colorMedium[3]{ 100,100,100 };
    bool contarst[3]{ 0,0,0 };
    bool writeCoordinats{ true };

    std::ofstream objectCoordinate("realObjectCoordinate.txt", std::ios_base::out | std::ios_base::trunc);
    for (; N > 0; --N)
    {
        int objColorMedium[3]{ 0,0,0 };
        cv::Mat outputImage(imageSize, imageSize, CV_8UC1);
        if (isColorImg)
        {
            cv::cvtColor(outputImage, outputImage, cv::COLOR_GRAY2BGR);
        }
        cv::Mat background(outputImage.size(), outputImage.type());
        bacgroundGenerate(background, objColorMedium, 1.2, 5, colorMedium, contarst);
        if (figureType != 0)
        {
            int sideSize{ random(42,35) };
            int diagonal{ static_cast<int>(pow(2 * sideSize * sideSize,0.5)) };
            int upLimit{ imageSize - diagonal / 2 };
            int downLimit{ diagonal / 2 };
            if (figureType == 1 || figureType == 3)
            {
                upLimit = imageSize - diagonal / 2;
                downLimit = diagonal / 2;
            }
            else if (figureType == 2)
            {
                upLimit = imageSize - sideSize/2;
                downLimit = sideSize/2;
            }
            cv::Point centerPoint(random(upLimit, downLimit), random(upLimit, downLimit));

            cv::Point vertices[4];
            if (figureType == 1 || figureType == 3)
            {
                float rotateAngel{ static_cast<float>((random(359))) };
                cv::RotatedRect rotRect(centerPoint, cv::Size(sideSize, sideSize), rotateAngel);
                cv::Point2f vertices2f[4];
                rotRect.points(vertices2f);
                for (int i{ 0 }; i < 4; ++i)
                {
                    vertices[i] = vertices2f[i];
                }
            }
            
            float width{ 0.0 };
            float heigth{ 0.0 };
            cv::Mat imgMask(outputImage.size(), outputImage.type(), cv::Scalar(0, 0, 0));
            if (figureType == 1)
            {
                cv::fillConvexPoly(imgMask, vertices, 4,cv::Scalar(255,255,255), 8);
                if(writeCoordinats)
                    findHeigtAndWeigth(vertices, 4, heigth, width);
            }
            else if (figureType == 2)
            {
                cv::circle(imgMask, centerPoint, sideSize/2, cv::Scalar(1, 1, 1), -1, -1);
                heigth = sideSize;
                width = sideSize;
            }
            else if (figureType == 3)
            {
                cv::fillConvexPoly(imgMask, vertices, 3, cv::Scalar(1, 1, 1), 8);
                if (writeCoordinats)
                    findHeigtAndWeigth(vertices, 3, heigth, width);
            }
            
            if (writeCoordinats)
            {
                objectCoordinate << figureType << " " << static_cast<float>(centerPoint.x) / imageSize << " " << static_cast<float>(centerPoint.y) / imageSize << " " <<
                                    static_cast<float>(heigth) / imageSize << " " << static_cast<float>(width) / imageSize << std::endl;
            }

            cv::Mat imgWithOject(objectGenerate(imgMask, 3, objColorMedium));
            cv::bitwise_not(imgMask, imgMask);
            cv::bitwise_and(background, imgMask, background);
            cv::bitwise_or(imgWithOject, background, outputImage);
        }
       
        cv::imwrite("train/true/t_" + std::to_string(N+20) + ".png", outputImage);
        std::cout << N << std::endl;
    }
    objectCoordinate.close();
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
