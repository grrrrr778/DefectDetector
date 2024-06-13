// File: ArgParser.h
// Purpose: parser of console (stdio) program arguments
// Authors: S. Korobkova, P. Skribtsov
// Date: Jan 2009
// (C) PAWLIN TECHNOLOGIES LTD. ALL RIGHTS RESERVED

#pragma once

#include <vector>
#include <string>
#include <map>
#include <PWNGeneral/pwnutil.h>
//#include <tchar.h>

class ArgParser {
	int arg_num;
	char** args;
	bool sensitive;

public:
	void getHangingArgs(std::vector<std::string> &out) const {
		for (int i = 1; i < arg_num; i++) {
			std::string val = getArg(i);
			if (val.size() && val[0] == '-') continue;
			if (val.find("=") == val.npos) out.push_back(val); // hanging means has no = sign and no -
		}
	}
	std::string getArg(size_t index) const {
		if ((int)index >= arg_num) return std::string();
		return std::string(args[index]);
	}
	int getInt(const char *paramname, int default_value) const {
		int v = default_value;
		queryInt(paramname, v);
		return v;
	}
	float getFloat(const char *paramname, float default_value) const {
		float v = default_value;
		queryFloat(paramname, v);
		return v;
	}
	std::string getString(const char *paramname, const std::string &default_value) const {
		auto v = default_value;
		queryString(paramname, v);
		return v;
	}
	template <typename T>
	T getXYZ(const char *paramname, const T &default_value) const {
		std::string rotation;
		if (!queryString(paramname, rotation)) return default_value;
		std::vector<std::string> parts;
		::split(rotation, ',', parts);
		assertEqualSize(parts.size(), 3, std::string("parameter ")+paramname+" must have 3 components such as 1,0,0");
		return { str2float(parts[0]),str2float(parts[1]),str2float(parts[2]) };
	}
	ArgParser(){arg_num=0; args=NULL;}
	ArgParser(int argc, char* argv[], bool set_sensitive = true) : arg_num(argc), args(argv),sensitive(set_sensitive) {};
	void queryAll(std::map<std::string,std::string> &Params) const; // fills all param=value pairs
	bool queryFlag(const char* flag) const;
	bool queryString(const char* paramName, std::string &value) const;
	bool queryDouble(const char* paramName, double &value) const;
	bool queryFloat(const char* paramName, float &value) const;
	bool querySize_t(const char *paramName, size_t &value) const {
		double d = 0;
		if(!queryDouble(paramName,d)) return false;
		value = (size_t) d;
		return true;
	}
	bool queryInt(const char *paramName, int &value) const {
		double d = 0;
		if(!queryDouble(paramName,d)) return false;
		value = (int) d;
		return true;
	}
	bool queryFilename(const char* paramName, std::string &value) const;
	void setCaseSensitive(bool flag = true);
	static bool readFilelist(const std::string &filelist, std::vector <std::string> &files);
	void saveToFile(const string &filename)const {
		FILE *file = fopen(filename.c_str(), "wt");
		saveToFile(file);
		fclose(file);
	}
	void saveToFile(FILE *fp)const ;
	void loadFromFile(FILE *fp) ;
	void loadFromBatFile(const std::string &filePath);
	void dumpToMap(std::map <std::string,std::string> &dump)const;
};