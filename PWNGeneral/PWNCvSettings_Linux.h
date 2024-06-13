#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <map>
#include <vector>
#include <PWNGeneral/IPWNSettings.h>
#include <PWNGeneral/PWNSettingsConversion.h>
#include <PWNGeneral/PWNExceptionBasic.h>


using namespace std;

enum PWNCvSettingsOpenMode {
	READ   = 0,
	WRITE  = 1,
	APPEND = 2

};



typedef void(*IOParamFunc)(void* settings, void *obj);
class PWNCvSettings :public IPWNSettings<cv::FileStorage> {
public:
	static std::map<std::string, std::map<std::string, void*>>saveParamFunctions;
	static std::map<std::string, std::map<std::string, void*>>loadParamFunctions;
private:
	cv::FileStorage storage;
	PWNCvSettingsOpenMode mode;
	std::string splitChar;
protected:

	template <typename T> 
		void addElement(const std::string &elementName, const T& value) {
			std::string str;
			PWNParamToString<T>(value, str);
			storage << elementName << str;	
		}
	
	template <typename T> bool getElement(const std::string &elementName, T& value)const {
		std::string str;
		cv::FileNode node = storage[elementName];
		if (!node.isNone()) {
			node >> str;
			PWNParamFromString<T>(str, value);
			return true;
		}
		else {
			printf(("tag " + elementName + " not exist in file\n").c_str());
			return false;
		}
	}
	
	
	
	/*--------specializations---------*/
	template <typename T> void addElementCv(const std::string &elementName, const T &value) 
	{
		storage << elementName << value;	
	}
	template <typename T> bool getElementCv(const std::string &elementName, T &value)const {
		cv::FileNode node = storage[elementName];
		if (!node.isNone())
			node >> value;
		else {
			printf(("tag " + elementName + " not exist in file\n").c_str());
			return false;
		}
		return true;
	}
	template <typename T> bool getElementCvString(const std::string &elementName, T &value)const {
			cv::FileNode node = storage[elementName];
			if (!node.isNone())
			{
				// if T is Sting but we need read Int type too		
				if(node.isInt()) {
					int i_int = node.operator int();
					std::string vvalue = std::to_string(i_int);
					value = vvalue;
				}
				else {
					node >> value;
				}
			}
			else {
				printf(("tag " + elementName + " not exist in file\n").c_str());
				return false;
			}
			return true;
		}
	
public:
	PWNCvSettings() {
		splitChar = "_";
	}
	bool open(const std::string &storageName, PWNCvSettingsOpenMode openMode) {
		mode = openMode;
		switch (mode) {
		case READ: {
				storage.open(storageName, cv::FileStorage::READ);
			}break;
		case WRITE: {
				storage.open(storageName, cv::FileStorage::WRITE);
			}break;
		case APPEND: {
				storage.open(storageName, cv::FileStorage::APPEND);
			}break;
		}
		;
		return storage.isOpened();
	}
	
	template<typename T> void addParam(const std::string &paramName, const std::vector<T> &paramValue, const std::string &nodeName = "") {
		if ((mode == PWNCvSettingsOpenMode::WRITE) || (mode == PWNCvSettingsOpenMode::APPEND)) {
			addElement(nodeName + splitChar + paramName, (int)paramValue.size());
			for (int k = 0; k < paramValue.size(); k++) {
				addElement(nodeName + splitChar + paramName + splitChar + to_string(k), paramValue[k]);
			}
		}
		else
			throw PWNExceptionBasic("Settings file was opened for writing");
	}
	template<typename ParamType>  void addParam(const std::string &paramName, const ParamType &paramValue, const std::string &nodeName = "") {
		if ((mode == PWNCvSettingsOpenMode::WRITE) || (mode == PWNCvSettingsOpenMode::APPEND))
			addElement(nodeName + splitChar + paramName, paramValue);
		else
			throw PWNExceptionBasic("Settings file was opened for writing");
	}

	template<typename T> void getParam(const std::string &paramName, std::vector<T> &paramValue, const std::string &nodeName = "")const {
		if (mode == PWNCvSettingsOpenMode::READ) {
			paramValue.clear();
			int size;
			if (getElement(nodeName + splitChar + paramName, size)) {
				for (int k = 0; k < size; k++) {
					T value;
					getElement(nodeName + splitChar + paramName + splitChar + to_string(k), value);
					paramValue.push_back(value);
				}
			}
		}
		else
			throw PWNExceptionBasic("Settings file was opened for reading");
	}
	template<typename ParamType> void getParam(const std::string &paramName, ParamType &paramValue, const std::string &nodeName = "")const {
		if (mode == PWNCvSettingsOpenMode::READ) {
			getElement(nodeName + splitChar + paramName, paramValue);
		}
		else
			throw PWNExceptionBasic("Settings file was opened for reading");
	}

	void release() {
		if (storage.isOpened())
			storage.release();
	}

};
/*-------add specialisations-----------*/
#ifdef WIN32
template <> void PWNCvSettings::addElement(const std::string &elementName, const unsigned __int64 &value) {
	int val = int(value);
	addElementCv(elementName, val);
}
#endif
template <> void PWNCvSettings::addElement<unsigned>(const std::string &elementName, const unsigned &value);
template <> void PWNCvSettings::addElement<bool>(const std::string &elementName, const bool &value);
template <> void PWNCvSettings::addElement<int>(const std::string &elementName, const int &value);
template <> void PWNCvSettings::addElement<float>(const std::string &elementName, const float &value);
template <> void PWNCvSettings::addElement<double>(const std::string &elementName, const double &value);
template <> void PWNCvSettings::addElement<std::string>(const std::string &elementName, const std::string &value);
template <> void PWNCvSettings::addElement<cv::Range>(const std::string &elementName, const cv::Range &value);
template <> void  PWNCvSettings::addElement< cv::Mat>(const std::string &elementName, const cv::Mat &value);
template <> void  PWNCvSettings::addElement<cv::SparseMat>(const std::string &elementName, const cv::SparseMat &value);
/*-------get specialisations-----------*/
#ifdef WIN32
template <> bool  PWNCvSettings::getElement(const std::string &elementName, unsigned __int64 &value)const {
	int val;
	if (getElementCv(elementName, val)) {
		value = size_t(val);
		return true;
	}
	else
		return false;

}
#endif
template <> bool  PWNCvSettings::getElement<unsigned>(const std::string &elementName, unsigned &value) const;
template <> bool  PWNCvSettings::getElement<bool>(const std::string &elementName, bool &value)const;
template <> bool  PWNCvSettings::getElement<int>(const std::string &elementName, int &value) const;
template <> bool  PWNCvSettings::getElement<float>(const std::string &elementName, float &value)const;
template <> bool  PWNCvSettings::getElement<double>(const std::string &elementName, double &value)const;

template <> bool  PWNCvSettings::getElement<std::string>(const std::string &elementName, std::string &value)const;
template <> bool  PWNCvSettings::getElement<cv::Range>(const std::string &elementName, cv::Range &value) const;
template <> bool  PWNCvSettings::getElement<cv::Mat>(const std::string &elementName, cv::Mat &value) const;
template <> bool PWNCvSettings::getElement<cv::SparseMat>(const std::string &elementName, cv::SparseMat &value) const;
/*------------------*/

#define IPWNCvSettingsSaveParam(PARAM) settings.addParam(#PARAM,PARAM,nodeName);
#define IPWNCvSettingsLoadParam(PARAM) settings.getParam(#PARAM,PARAM,nodeName);



class  IPWNCvSettingsSaveable {
protected:
	std::string nodeName;
public:
	IPWNCvSettingsSaveable(std::string nodeName)
		: nodeName(nodeName) {

	}
	virtual void save(PWNCvSettings &settings)const = 0;
	void save(const std::string &filename)const {
		PWNCvSettings settings;
		settings.open(filename, PWNCvSettingsOpenMode::WRITE);
		save(settings);
		settings.release();
	}
	virtual void load(PWNCvSettings &settings) = 0;
	void load(const std::string &filename)
	{
		PWNCvSettings settings;
		settings.open(filename, PWNCvSettingsOpenMode::READ);
		load(settings);
		settings.release();
	}
	/*void save(PWNCvSettings &settings,const std::string className){
		for(int k=0;k<PWNCvSettings::saveParamFunctions[className].size();k++){
			PWNCvSettings::saveParamFunctions[className][k](&settings,this);

}

}*/
};




#define DECLARE_PARAM(PARAM_TYPE,PARAM_NAME,CLASS_NAME)         PARAM_TYPE PARAM_NAME;                \
	static void load##PARAM_NAME(void *settings,void *obj){                                              \
		CLASS_NAME* ptr=(CLASS_NAME*)obj;                                                              \
		PWNCvSettings* settingsPtr=(PWNCvSettings*)settings;                                           \
		settingsPtr->getParam(#PARAM_NAME,ptr->PARAM_NAME,#CLASS_NAME);   \
	};   \
	static void save##PARAM_NAME(void *settings,void *obj){                                              \
		CLASS_NAME* ptr=(CLASS_NAME*)obj;                                                              \
		PWNCvSettings* settingsPtr=(PWNCvSettings*)settings;                                           \
		settingsPtr->addParam(#PARAM_NAME,ptr->PARAM_NAME,#CLASS_NAME);   \
	};                                                                                 \
struct __reg##PARAM_NAME{	                                                                  \
 __reg##PARAM_NAME(){                                                                         \
		PWNCvSettings::saveParamFunctions[#CLASS_NAME][#PARAM_NAME]=(void*)&CLASS_NAME::save##PARAM_NAME;           \
		PWNCvSettings::loadParamFunctions[#CLASS_NAME][#PARAM_NAME]=(void*)&CLASS_NAME::load##PARAM_NAME;			\
		}                                                                                               \
};                                                                                                      \
__reg##PARAM_NAME reg##PARAM_NAME##_obj;	\

#define DECLARE_PARAM_STRUCT(STRUCT_TYPE,STRUCT_NAME,CLASS_NAME)         STRUCT_TYPE STRUCT_NAME;                \
	static void load##STRUCT_NAME(void *settings,void *obj){                                              \
		CLASS_NAME* ptr=(CLASS_NAME*)obj;                                                              \
		PWNCvSettings* settingsPtr=(PWNCvSettings*)settings;                                           \
		ptr->STRUCT_NAME.load(*settingsPtr);   \
	};   \
	static void save##STRUCT_NAME(void *settings,void *obj){                                              \
		CLASS_NAME* ptr=(CLASS_NAME*)obj;                                                              \
		PWNCvSettings* settingsPtr=(PWNCvSettings*)settings;                                           \
		ptr->STRUCT_NAME.save(*settingsPtr);   \
	};                                                                                 \
struct __reg##STRUCT_NAME{	                                                                  \
 __reg##STRUCT_NAME(){                                                                         \
		PWNCvSettings::saveParamFunctions[#CLASS_NAME][#STRUCT_NAME]=(void*)&CLASS_NAME::save##STRUCT_NAME;           \
		PWNCvSettings::loadParamFunctions[#CLASS_NAME][#STRUCT_NAME]=(void*)&CLASS_NAME::load##STRUCT_NAME;			\
		}                                                                                               \
};                                                                                                      \
__reg##STRUCT_NAME reg##STRUCT_NAME##_obj;	\

#define PARAM_SAVELOAD(CLASS_NAME)                                                 \
void save(PWNCvSettings &settings)const{                                           \
																					\
	for(map<string,void*>::iterator iter=PWNCvSettings::saveParamFunctions[#CLASS_NAME].begin();iter!=PWNCvSettings::saveParamFunctions[#CLASS_NAME].end();iter++){                    \
		IOParamFunc func=(IOParamFunc)iter->second;  		\
	     func((void*)&settings,(void*)this);                                                         \
	}                                                                                       \
}                                                                                    \
void save(const std::string &filename)const{                                         \
		PWNCvSettings settings;                                                      \
		settings.open(filename,PWNCvSettingsOpenMode::WRITE);                        \
		save(settings);                                                             \
		settings.release();                                                          \
	}                                                                                \
void load(PWNCvSettings &settings)const{                                           \
																					\
	for(map<string,void*>::iterator iter=PWNCvSettings::loadParamFunctions[#CLASS_NAME].begin();iter!=PWNCvSettings::loadParamFunctions[#CLASS_NAME].end();iter++){                   \
		IOParamFunc func=(IOParamFunc)iter->second;  		\
	     func((void*)&settings,(void*)this);                                                         \
	}                                                                                       \
}                                                                                    \
void load(const std::string &filename)const{                                         \
		PWNCvSettings settings;                                                      \
		settings.open(filename,PWNCvSettingsOpenMode::READ);                        \
		load(settings);                                                             \
		settings.release();                                                          \
	}    \
	/*		void *ptr=PWNCvSettings::saveParamFunctions[#CLASSNAME][k];					\
saveParamFunc func=(saveParamFunc)ptr;     \
func((void*)&settings,(void*)this);           \*/

