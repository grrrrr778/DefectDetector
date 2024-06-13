#include "stdafx.h"
#include <PWNGeneral/PWNCvSettings_Linux.h>

template <> void PWNCvSettings::addElement<unsigned>(const std::string &elementName, const unsigned &value) {
	int val = int(value);
	addElementCv(elementName, val);
}
template <> void PWNCvSettings::addElement<bool>(const std::string &elementName, const bool &value) {
	addElementCv(elementName, value);
}
template <> void PWNCvSettings::addElement<int>(const std::string &elementName, const int &value) {
	addElementCv(elementName, value);
}
template <> void PWNCvSettings::addElement<float>(const std::string &elementName, const float &value) {
	addElementCv(elementName, value);
}
template <> void PWNCvSettings::addElement<double>(const std::string &elementName, const double &value) {
	addElementCv(elementName, value);
} 
template <> void PWNCvSettings::addElement<std::string>(const std::string &elementName, const std::string &value) {
	addElementCv(elementName, value);
}
template <> void PWNCvSettings::addElement<cv::Range>(const std::string &elementName, const cv::Range &value) {
	addElementCv(elementName, value);
}
template <> void  PWNCvSettings::addElement< cv::Mat>(const std::string &elementName, const cv::Mat &value) {
	addElementCv(elementName, value);
}
template <> void  PWNCvSettings::addElement<cv::SparseMat>(const std::string &elementName, const cv::SparseMat &value) {
	addElementCv(elementName, value);
}

template <> bool  PWNCvSettings::getElement<unsigned>(const std::string &elementName, unsigned &value)const {
	int val;
	if (getElementCv(elementName, val)) {
		value = size_t(val);
		return true;
	}
	else {
		return false;
	}
}

template <> bool  PWNCvSettings::getElement<bool>(const std::string &elementName, bool &value)const {
	if (getElementCv(elementName, value))
		return true;
	else
		return false;
}
template <> bool  PWNCvSettings::getElement<int>(const std::string &elementName, int &value)const {
	if (getElementCv(elementName, value))
		return true;
	else
		return false;
}
template <> bool  PWNCvSettings::getElement<float>(const std::string &elementName, float &value)const {
	if (getElementCv(elementName, value))
		return true;
	else
		return false;
}
template <> bool  PWNCvSettings::getElement<double>(const std::string &elementName, double &value)const {
	if (getElementCv(elementName, value))
		return true;
	else
		return false;
}

template <> bool  PWNCvSettings::getElement<std::string>(const std::string &elementName, std::string &value)const {
	if (getElementCvString(elementName, value))
		return true;
	else
		return false;
}
template <> bool  PWNCvSettings::getElement<cv::Range>(const std::string &elementName, cv::Range &value)const {
	if (getElementCv(elementName, value))
		return true;
	else
		return false;
}
template <> bool  PWNCvSettings::getElement<cv::Mat>(const std::string &elementName, cv::Mat &value)const {
	if (getElementCv(elementName, value))
		return true;
	else
		return false;
}
template <> bool PWNCvSettings::getElement<cv::SparseMat>(const std::string &elementName, cv::SparseMat &value)const {
	if (getElementCv(elementName, value))
		return true;
	else
		return false;
}
/*------------------*/

std::map<std::string, map<string, void*>> PWNCvSettings::saveParamFunctions = map<string, map<string, void*>>();
map<std::string, map<string, void*>> PWNCvSettings::loadParamFunctions = map<string, map<string, void*>>();
