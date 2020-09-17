//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#include "pch.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>  // cv::Canny()
#include <opencv2/aruco.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/video/tracking.hpp>
#include "OpenCVFrameProcessing.h"


void ProcessRmFrameWithOpticalFlow(IResearchModeSensorFrame* pSensorFrame, cv::Mat& cvResultMat, std::vector<int>& ids, std::vector<std::vector<cv::Point2f>>& corners, FrameCache& kFrameCache)
{
    HRESULT hr = S_OK;
    ResearchModeSensorResolution resolution;
    size_t outBufferCount = 0;
    const BYTE* pImage = nullptr;
    IResearchModeSensorVLCFrame* pVLCFrame = nullptr;
    pSensorFrame->GetResolution(&resolution);

    hr = pSensorFrame->QueryInterface(IID_PPV_ARGS(&pVLCFrame));
    if (SUCCEEDED(hr))
    {
        pVLCFrame->GetBuffer(&pImage, &outBufferCount);

        cv::Mat processed(resolution.Height, resolution.Width, CV_8U, (void*)pImage);

        if (kFrameCache.m_bInitialized == false)
        {            

            cv::Mat mask = cv::Mat::zeros(processed.size(), processed.type());
            cv::goodFeaturesToTrack(processed, kFrameCache.m_vecPreviousGoodPoints, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);
          
            //cache and raise initialized flag
            mask.copyTo(kFrameCache.m_matDrawMask);
            processed.copyTo(kFrameCache.m_matPreviousGrey);
            kFrameCache.m_bInitialized = true;
        }
        else
        {
            std::vector<uchar> status;
            std::vector<float> err;
            cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 10, 0.03);
            std::vector<cv::Point2f> tempGoodPoints;
            std::vector<cv::Point2f> newGoodPoints;
            cv::calcOpticalFlowPyrLK(kFrameCache.m_matPreviousGrey, processed, kFrameCache.m_vecPreviousGoodPoints, tempGoodPoints, status, err, cv::Size(15, 15), 2, criteria);
             
            for (uint i = 0; i < kFrameCache.m_vecPreviousGoodPoints.size(); i++)
            {
                if (status[i] == 1)
                {
                    newGoodPoints.push_back(tempGoodPoints[i]);
                    cv::line(kFrameCache.m_matDrawMask, tempGoodPoints[i], kFrameCache.m_vecPreviousGoodPoints[i], kFrameCache.m_vecColors[i], 2);
                    //cv::circle(processed, tempGoodPoints[i], 5, kFrameCache.m_vecColors[i], -1);
                }
            }
            
            //cache new frame and goodtpoints for next frame
            processed.copyTo(kFrameCache.m_matPreviousGrey);
            kFrameCache.m_vecPreviousGoodPoints = newGoodPoints;

            //draw line after caching for visual purpose
            cv::add(kFrameCache.m_matDrawMask, processed, processed);

        }

        cvResultMat = processed;
    }

    if (pVLCFrame)
    {
        pVLCFrame->Release();
    }
}

void ProcessRmFrameWithAruco(IResearchModeSensorFrame* pSensorFrame, cv::Mat& cvResultMat, std::vector<int> &ids, std::vector<std::vector<cv::Point2f>> &corners, FrameCache& kFrameCache)
{
    HRESULT hr = S_OK;
    ResearchModeSensorResolution resolution;
    size_t outBufferCount = 0;
    const BYTE *pImage = nullptr;
    IResearchModeSensorVLCFrame *pVLCFrame = nullptr;
    pSensorFrame->GetResolution(&resolution);
    //static cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    hr = pSensorFrame->QueryInterface(IID_PPV_ARGS(&pVLCFrame));

    if (SUCCEEDED(hr))
    {
        pVLCFrame->GetBuffer(&pImage, &outBufferCount);

        cv::Mat processed(resolution.Height, resolution.Width, CV_8U, (void*)pImage);
        //cv::aruco::detectMarkers(processed, dictionary, corners, ids);

        cvResultMat = processed;

        // if at least one marker detected
        //if (ids.size() > 0)
          //  cv::aruco::drawDetectedMarkers(cvResultMat, corners, ids);
    }

    if (pVLCFrame)
    {
        pVLCFrame->Release();
    }
}


void ProcessRmFrameWithCanny(IResearchModeSensorFrame* pSensorFrame, cv::Mat& cvResultMat)
{
    HRESULT hr = S_OK;
    ResearchModeSensorResolution resolution;
    size_t outBufferCount = 0;
    const BYTE *pImage = nullptr;
    IResearchModeSensorVLCFrame *pVLCFrame = nullptr;
    pSensorFrame->GetResolution(&resolution);

    hr = pSensorFrame->QueryInterface(IID_PPV_ARGS(&pVLCFrame));

    if (SUCCEEDED(hr))
    {
        pVLCFrame->GetBuffer(&pImage, &outBufferCount);

        cv::Mat processed(resolution.Height, resolution.Width, CV_8U, (void*)pImage);

        cv::Canny(processed, cvResultMat, 400, 1000, 5);
    }

    if (pVLCFrame)
    {
        pVLCFrame->Release();
    }
}


