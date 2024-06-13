// (c) Pawlin Technologies Ltd. 2008
// Pawlin Utility Library
// contains common algorithms useful in different software
// ansi-C/C++ based
// Author: Pavel Skribtsov, Alex Dolgopolov, Sergey Zagoruiko
// Version: 1.0
// Modify date: 10-JUN-08

#pragma once

#include <stdint.h>
#include <stdlib.h>

#include <string>
#include <string.h>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
//#include "pwnwinutil.h" // don't include this here - it mixes cross platform and windows specific code!
//#ifndef WIN32
//#endif
#include <stdio.h>

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <mutex>

#include <sys/stat.h>

#ifndef M_PIf
#define M_PIf 3.14159265f
#endif

// sorry guys, this macro is by lazy skribtsov
// (I decided to introduce it after 20 years of programming in C++)
// ...what's interesting  
// ... new C++ standard has now similar new sytax for such cycle!!! .. something like "for(auto i : array)" , so this means I am in the trend :-)
#define FOR_ALL(vname,i) for(size_t i = 0; i < vname.size(); i++)
#define FOR_ALL_INT(vname,i) for(int i = 0; i < (int) vname.size(); i++)
#define FOR_ALL_IF(vname,xx,condition) for(size_t xx = 0; xx < vname.size(); xx++) if(condition)
inline const char *yesno(bool v) { return v ? "YES" : "NO"; }

template <typename Tz>
const char *endcomma(const Tz &array, size_t n) { return n + 1 == array.size() ? "\n" : ","; }
inline float rnd() { return rand() / (float)RAND_MAX; }
inline float rnd(float minv, float maxv) { return minv + rnd()*(maxv-minv); }
#define escape_ascii 27
#ifndef WIN32
	#define _cls printf("%c[2J",escape_ascii)
	#define set_cursor(row,col) printf("%c[%d;%dH",escape_ascii,row,col)
#else
	#define _cls system("cls")
	#define set_cursor(row,col) {COORD pos = {col, row}; SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), pos);}
#endif


using std::string;
using std::vector;
int getch_os();
void delaySeconds(float t) ;
int kbhit_os();

#ifdef _MSC_VER
	#pragma warning(disable : 4996)
	#define finite_check _finite
	//#include <math.h>
	//#include <cmath>

	//#pragma warning(disable : 4996)
	//#define finite_check _finite
	//#define finite_check std::isfinite
#else
#ifndef _ISOC99_SOURCE
#define _ISOC99_SOURCE
#endif
#ifndef __USE_ISOC99
#define __USE_ISOC99
#endif

#include <math.h>

//#ifndef finite_check
	#include <cmath>
	#define finite_check std::isfinite
//#else
//	#define finite_check isfinite
//#endif
#endif 



#ifdef _MSC_VER
typedef unsigned __int64 uint64_t;
typedef unsigned __int32 uint32_t;
typedef __int64 int64_t;
#endif

#ifndef MAX_PATH
#ifdef  PATH_MAX
#define MAX_PATH PATH_MAX
#else
#define MAX_PATH 260
#endif
#endif

#if (defined(WIN32) || defined(_WIN64)) && !defined __MINGW32__
#include <Windows.h>
#include <PWNAudio/dirent.h>
#endif
#ifndef WIN32
#ifndef STM32
#include "dirent.h"
#endif
#endif

#ifndef float2int
inline int float2int(float x) {
	if (x < 0) return int(x - 0.5f);
	return int(x + 0.5f);
}
#endif
inline float str2float(const string &str) { return (float)atof(str.c_str()); }
inline void save(FILE *file, const string &str) {
		uint64_t isize = str.size();
		fwrite(&isize,sizeof(isize),1,file);
		if(isize!=0)
			fwrite(str.c_str(),sizeof(char), (size_t) isize,file);
}
inline void load(FILE *file, std::string &array) {
		uint64_t isize = 0;
		fread(&isize,sizeof(isize),1,file);
		array.resize((size_t)isize);
		if(isize!=0)
			fread(&(array[0]),sizeof(char), (size_t)isize,file);
}

inline void save(FILE *file, const vector <string> &array) {
	uint64_t isize = array.size();
	fwrite(&isize, sizeof(isize), 1, file);
	FOR_ALL(array, i) save(file, array[i]);
}

inline void load(FILE *file, vector <string> &array) {
	uint64_t isize = 0;
	fread(&isize, sizeof(isize), 1, file);
	array.resize((size_t) isize);
	FOR_ALL(array, i) load(file, array[i]);
}



template <class _T> 
class MapFileSaver { // required _T to have .save / load (file) methods
public:
	static void save(FILE *file, const std::map <std::string,_T> &array) {
		uint64_t isize = array.size();
		fwrite(&isize,sizeof(isize),1,file);
		for(typename std::map <std::string,_T>::const_iterator iter = array.begin(); iter != array.end(); ++iter) {
			::save(file,iter->first);
			iter->second.save(file);
		}
	}
	static void load(FILE *file, std::map <std::string,_T> &array) {
		uint64_t isize = 0;
		fread(&isize,sizeof(isize),1,file);
		for(size_t i = 0; i < isize; i++) {
			std::string str;
			::load(file,str);
			_T obj;
			obj.load(file);
			array[str] = obj;
		}
	}
};
template <class _T> 
class FileSaver {
public:
	static void save(const string &fname, const _T& obj) {
		save(fname, vector<_T>(1, obj));
	}
	static void load(const string &fname, _T& obj) {
		vector<_T> array;
		load(fname, array);
		if (array.size() != 1) throw std::runtime_error("FileSaver::load single object failed - count of objects!=1");
		obj = array.front();
	}
	static void save(FILE *file, const std::vector <_T> &array) {
			uint64_t isize = array.size();
			fwrite(&isize,sizeof(isize),1,file);
			if(isize!=0)
				fwrite(&array.front(),sizeof(_T), (size_t) isize,file);
	}
	static void load(FILE *file, std::vector <_T> &array) {
			uint64_t isize = 0;
			fread(&isize,sizeof(isize),1,file);
			array.resize((size_t)isize);
			if(isize!=0)
				fread(&array.front(),sizeof(_T), (size_t)isize,file);
	}

	static void save(FILE *file, const std::vector < std::vector <_T > > &graph) {
		uint64_t size = graph.size();
		fwrite(&size,sizeof(size),1,file);
		for(size_t i = 0; i < size; i++) save(file,graph[i]);
	}
	static void load(FILE *file, std::vector < std::vector < _T > > &graph) {
		uint64_t size = 0;
		fread(&size,sizeof(size),1,file);
		graph.resize((size_t)size);
		for(size_t i = 0; i < size; i++) load(file,graph[i]);
	}

	static void save(const string &filename, const std::vector <_T> &array) {
		save(filename.c_str(), array);
	}
	static void save(const char * filename, const std::vector <_T> &array)
	{
		FILE * file = NULL;
		file = fopen( filename, "wb");
		if ( NULL == file)
		{
			throw "Couldn't open file for writing";
		}
		save(file, array);

		fclose(file);
	}
	static void load(const string & filename, std::vector <_T> &array) {
		load(filename.c_str(), array);
	}

	static void load(const char * filename, std::vector <_T> &array) {
		FILE * file = NULL;
		file = fopen( filename, "rb" );
		if ( NULL == file )
		{
			throw std::runtime_error("Couldn't open file for reading:"+string(filename));
		}
		load(file, array);

		fclose(file);
	}

	static void save(const string & filename, const std::vector < std::vector <_T > > &graph) {
		save(filename.c_str(), graph);
	}
	static void save(const char * filename, const std::vector < std::vector <_T > > &graph) {
		FILE * file = NULL;
		file = fopen( filename, "wb");
		if ( NULL == file)
		{
			throw "Couldn't open file for writing";
		}
		save(file, graph);

		fclose(file);
	}
	static void load(const string & filename, std::vector < std::vector < _T > > &graph) {
		load(filename.c_str(), graph);
	}
	static void load(const char * filename, std::vector < std::vector < _T > > &graph) {
		FILE * file = NULL;
		file = fopen( filename, "rb" );
		if ( NULL == file )
		{
			throw "Couldn't open file for reading";
		}
		load(file, graph);		

		fclose(file);
	}
};

template <class _T> 
class FileSaverUsingSaveLoad {
public:
	static void save(FILE *file, const std::vector<_T> &array) {
		uint64_t size = array.size();
		fwrite(&size,sizeof(size),1,file);
		for(size_t i = 0; i < array.size(); i++) array[i].save(file);
	}

	static void load(FILE *file, std::vector<_T> &array) {
		uint64_t size = 0;
		fread(&size,sizeof(size),1,file);
		array.resize((size_t) size);
		for(size_t i = 0; i < size; i++) array[i].load(file);
	}

	static void save(const char * filename, const std::vector <_T> &array)
	{
		FILE * file = NULL;
		file = fopen( filename, "wb");
		if ( NULL == file)
		{
			throw "Couldn't open file for writing";
		}
		save(file, array);

		fclose(file);
	}

	static void load(const char * filename, std::vector <_T> &array) {
		FILE * file = NULL;
		file = fopen( filename, "rb" );
		if ( NULL == file )
		{
			throw "Couldn't open file for reading";
		}
		load(file, array);

		fclose(file);
	}

};
int writeFileName (const char *filename, void* userargs);

FILE* saveFileToTemp(const char * in_filename);//return pointer to tempfile

int writeFileNameOnlyInSubDir (const char *filename, void* userargs);

void  restoreFileFromTemp(FILE* temp, const char * out_filename);
#ifndef STM32
inline void pwnSleep( int ms ) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }
#endif
uint64_t makeID();
std::string makeDateTimeID();

//return -1, if file not exist, or can't open
long getFileSize(const char* path);

class SimpleIntParams {
protected:
	std::map <std::string, int> int_Params;
public:
	int getInt(const std::string &name) const;
	void loadFromTextFile(const char *filename);
};

class SimpleStringParams {
protected:
	std::map <std::string, std::string> string_Params;
public:
	const std::string &getString(const std::string &name) const;
	int getIntValue(const std::string &name) const { return atoi(getString(name).c_str());}
	float getFloatValue(const std::string &name) const { return (float)atof(getString(name).c_str());}
	void loadFromTextFile(const char *filename);
};
inline const char *bool2str(bool b) {
	static const char *yes = "YES";
	static const char *no = "NO";
	return b ? yes : no;
}
inline std::string int2str(int64_t id) {
	return std::to_string(id);
}
inline std::string int2str(int id) {
	return std::to_string(id);
}
inline std::pair<std::string, int> parseNameInt(const string &id) {
	std::string name; name.reserve(id.size());
	std::string num; num.reserve(id.size());
	for (char const &c : id) {
		if (isalpha(c)) name.push_back(c);
		else num.push_back(c);
	}
	return { name,atoi(num.c_str()) };
}
inline std::string leadZ(const string &str, size_t n_zero) {
	size_t n_zero_corr = n_zero < str.length() ? str.length() : n_zero;
	return std::string(n_zero_corr - str.length(), '0') + str;
}

inline std::string float2str(float v, const int digits=6) {
	char buf[64];
	sprintf(buf, "%.*f",digits, v); 
	return buf;
}

inline std::string float2str(double v, const int digits = 6) {
	char buf[128];
	sprintf(buf, "%.*lf", digits, v);
	return buf;
}
inline std::string smart_float2str(float v, const int digits=6) {
	int iv = int(v);
	if((float)iv==v) return int2str(iv);
	return float2str(v,digits);
}
inline std::string smart_float2str(double v, const int digits = 6) {
	int iv = int(v);
	if ((double)iv == v) return int2str(iv);
	return float2str(v, digits);
}
inline std::string int2str(::size_t id) {
	return std::to_string(id);
}
#define FixedNameMaxSize 256
struct FixedName {
	char name[FixedNameMaxSize];
	std::string getName() const {return name;}
	void setName(const std::string &str) {
		if(str.size()+1>FixedNameMaxSize) throw("too long name");
		strcpy(name,str.c_str());
	}
};

//---Macros Equivalents
template<typename T>
inline T sqr(const T & in) {
	return in*in;
}

struct BestElement {
	::size_t index;
	float metric;
	BestElement(::size_t i, float m) : index(i), metric(m) {}
	BestElement() {index = 0; metric = 0;}
	bool operator < (const BestElement &other) const
	{
		return metric < other.metric;
	}
};

template<typename T>
struct Range {
	T boundary_left;
	T boundary_right;

	Range(void) { }	
	Range(const T & boundary_left, const T & boundary_right) : boundary_left(boundary_left), boundary_right(boundary_right) {		
		if ( !(this->boundary_left <= this->boundary_right) ) throw "Incorrect boundaries for the range\n";
	}


	inline bool contain(const T & quantity) const { return ((quantity >= this->boundary_left) && (quantity <= this->boundary_right)); }

	inline T size(void) const { return this->boundary_right - this->boundary_left; }

};
typedef int FileCallback(const char *filename, void *userargs);

inline void addslashtopath(string &testpath) {
	if (testpath[testpath.size() - 1] != '\\' && testpath[testpath.size() - 1] != '/')
		testpath.push_back('\\');
}

inline void changeslash(string &str) {
	for (size_t k = 0; k<str.size(); k++) {
		if (str[k] == '/')str[k] = '\\';
	}
}
inline void inversechangeslash(string &str) {
	for (size_t k = 0; k<str.size(); k++) {
		if (str[k] == '\\')str[k] = '/';
	}
}
inline void changeslash(char *ptr) {
	while (*ptr) {
		if (*ptr == '/') *ptr = '\\';
		ptr++;
	}
}


class FileProcessor {
public:
	FileProcessor() {}
	virtual ~FileProcessor() {}
	void processPath(const char *path, const char *wildcard, bool verbose, bool skiphidden = false);
protected:
	virtual bool processFile(const char *filename) = 0; // return true if continue
private:
	static int fileCallback(const char *filename, void *userargs);
};

class FileList : public FileProcessor {
protected:
	vector <string> files;
public:
	FileList() {
	}
	void setFiles(const vector <string> & new_files) {
		files = new_files;
	}
	FileList(const string &path, const string &wildcard, bool random = false) {
		if (path.empty() || wildcard.empty()) return;
		FileProcessor::processPath(path.c_str(), wildcard.c_str(), false);
	/*	if (random)
			random_shuffle(files.begin(), files.end());*/
	}
	void sortByFolders(const vector <string> &folders, const vector<size_t> &indices) const;
	void sortByNumber(bool ascending = true);
	virtual bool processFile(const char *filename) {
		files.push_back(filename);
		return true;
	}
	size_t size() const {	return files.size();	}
	const vector <string> &getFiles() const { return files; }
	const string& operator [](size_t index) const { return files[index]; }
	void filter(const std::set<string> &exclusions) {
		auto rm_iter = std::remove_if(
			files.begin(), 
			files.end(), 
			[&](const string &s) { return (bool)exclusions.count(s); });
		files.erase(rm_iter, files.end());
	}
};

// this function was simply WRONG, I spent two ! night hours trying to figure out why my project didn't work!! what a shame!!!
// THIS IS WHAT IT WAS
// GAME: FIND A FEW(!) ERRORS:
//
//template <typename InputIt, typename ValueType, typename ContainerType>
//void split(InputIt begin, InputIt end, const ValueType &delimiter, std::vector<ContainerType> &out)
//{
//	out.clear();
//
//	ContainerType current_part;
//	while(begin != end)
//	{
//		if(*begin == delimiter)
//		{
//			if(!current_part.empty()) out.push_back(current_part);
//			continue;
//		}
//
//		current_part.push_back(*begin);
//		begin++;
//	}
//}

template <typename InputIt, typename ValueType, typename ContainerType>
void split(InputIt begin, InputIt end, const ValueType &delimiter, std::vector<ContainerType> &out)
{
	out.clear();

	ContainerType current_part;
	for(;begin != end;begin++)
	{
		if(*begin == delimiter) {
			if(!current_part.empty()) {
				out.push_back(current_part);
				current_part.clear();
			}
		}
		else
			current_part.push_back(*begin);
	}
	if(!current_part.empty()) out.push_back(current_part);
}
void split(const string &str,char separator,vector<string> &result, bool allow_empty = false);

void split(const string &str,vector<char> separator,vector<string> &result);



#if !defined(_WIN32) && !defined(__ANDROID__  ) && !defined(STM32) && !defined(__APPLE__)
#include <mcheck.h>
#include <iostream>

inline void checkMemory(const void * adr, const char * location)
{
	mcheck_status status_res = mprobe((void*)adr);

	if(status_res == MCHECK_OK)
		return;
	std::cout<<"Consistency at "<<location<<": ";
	switch(status_res)
	{
	case MCHECK_DISABLED:
		std::cout<<"MCHECK_DISABLED";
		break;
	case MCHECK_OK:
		std::cout<<"MCHECK_OK";
		break;
	case MCHECK_HEAD:
		std::cout<<"MCHECK_HEAD";
		break;
	case MCHECK_TAIL:
		std::cout<<"MCHECK_TAIL";
		break;
	case MCHECK_FREE:
		std::cout<<"MCHECK_FREE";
		break;
	default:
		std::cout<<"MCHECK_UNKNOWN";
		break;
	}
	std::cout<<std::endl;
}
#endif

bool pwnRandomPass(float probability /*0...1*/);

void findImgFiles(const string & path, vector<string> & img_files);

int onFindImgFile(const char* path, void* Params);
int getNumFromFilename(const string &file);
string getFileFromFullPath(const string &fullpath);
string removeExtensionFromFileName(const string & fname, string &fname_no_ext);
bool canOpen(const string &fname);
void removeFileNameFromPath(const string &fname, string &path);
void extractFileExtension(const string &fname, string &ext); // extracts extension without "."
std::string correct_path(const std::string & path);
std::string remove_folder_slash(const std::string & path);
inline bool check_dir_exist(const string &path) {
	struct stat sb;
	return stat(path.c_str(), &sb) == 0;
}
void create_dir(std::string path);
void copy_file(const std::string &file, const std::string &where);
void clean_dir(const std::string &wildcard);
void rename_dir(const std::string &parentDir, const std::string &nameFrom, const std::string &nameTo);

std::string get_cur_time(bool print = true);

std::string exec_command_with_msg_thru_pipe(const char* cmd);
std::string exec_command_with_msg_thru_file(const char* command);
std::string ZeroPadNumber(int num, int zeros_num = 7);

inline void assertEqualSize(size_t a, size_t b, const string &msg) {
	if (a != b) throw std::runtime_error(
		msg + " " +
		std::to_string(a) + "!=" + std::to_string(b));
}

inline void assertIndexRange(size_t sz, int v, const string &msg) {
	if(v < 0 || v >= int(sz)) throw std::runtime_error(
		msg + " " +
		std::to_string(v) + " not belong [0," + std::to_string(sz)+"]");
}

template <class T>
class RingBuffer {
	vector<T> storage;
	size_t head; // pointer to the end of queue
	size_t effective_size;
public:
	RingBuffer(size_t maxsize) : storage(maxsize) {
		effective_size = 0;
		head = 0;
	}
	bool push_back(const T &cf) {
		storage[head] = cf; // CamFrame assign operator should be called! -> no memory reallocation
		head++;
		head = head % storage.size();

		//std::cout << "head after push back: " << head << "\n";
		if (effective_size < storage.size()) {
			effective_size++;
			return true;
			//std::cout << "effective_size after push back: " << effective_size << "\n";
		}
		return false;
	}
	bool pop_front(T &frame) { // most old frame arrived
		if (effective_size == 0) return false;
		frame = front();
		effective_size--;
		return true;
	}
	bool pop() {
		//std::cout << "pop\n";
		if (effective_size == 0) return false;
		effective_size--;
		//std::cout << "effective_size = " << effective_size << "\n";
		return true;
	}
	size_t size() const { return effective_size; }
	bool empty() const { return size() == 0; }
	const T& front() const {
		if (effective_size == 0) throw std::runtime_error("empty CamFrameRingBuffer front() request!");
		size_t wherefrom_index = (head + storage.size() - effective_size) % storage.size();
		//std::cout << "front() wherefrom_index" << wherefrom_index << "\n";
		return storage[wherefrom_index];
	}

	void get_vector(std::vector<T>& v)
	{
		v.clear();
		for (size_t i = 0; i < effective_size; i++)
		{
			size_t wherefrom_index = (head + storage.size() - effective_size + i) % storage.size();
			v.push_back(storage[wherefrom_index]);
		}
		return;
	}

};
#if !defined(STM32)
class HeadedLogger {
	FILE *file = nullptr;
	bool first;
	// for delayed writing
	vector<uint64_t> timestamps;
	vector<vector<string> > datapoints;
	vector<string> header;
public:
#define STR_NAME(x) {x,#x}
#define VAL_NAME(x) {std::to_string(x),#x}
#define VAL_FLOAT(x) {float2str(x,8),#x}
	
	HeadedLogger() {}
	void open(const string &filename, size_t sz = 0) {
		file = fopen(filename.c_str(), "wt");
		if (!file) {
			::printf("HeadedLogger can't open file %s\n", filename.c_str());
			throw std::runtime_error("HeadedLogger can't open file " + filename);
		}
		first = true;
		timestamps.reserve(sz);
		datapoints.reserve(sz);
		header.reserve(sz);
	}
	HeadedLogger(const string &filename, size_t sz = 0) {
		open(filename);
		timestamps.reserve(sz);
		datapoints.reserve(sz);
		header.reserve(sz);
	}
	~HeadedLogger() {
		if(file) fclose(file);
	}
	void printf(const vector < std::pair<float, string>> &data) {
		if (first) {
			for (auto &iter : data) fprintf(file, "%s;", iter.second.c_str());
			fprintf(file, "\n");
			first = false;
		}
		for (auto &iter : data) fprintf(file, "%f;", iter.first);
		fprintf(file, "\n");
	}
	void printf(uint64_t ts, const vector < std::pair<float, string>> &data) {
		if (first) {
			fprintf(file, "%s;", "ts");
			for (auto &iter : data) fprintf(file, "%s;", iter.second.c_str());
			fprintf(file, "\n");
			first = false;
		}
		fprintf(file, "%s;", std::to_string(ts).c_str());
		for (auto &iter : data) fprintf(file, "%f;", iter.first);
		fprintf(file, "\n");
	}

	void printf_store(uint64_t ts, const vector < std::pair<string, string>> &data) {
		if (first) {
			header.push_back("ts");
			for (auto &iter : data) header.push_back(iter.second);
			first = false;
		}
		timestamps.push_back(ts);
		datapoints.push_back(vector<string>(data.size()));
		FOR_ALL(datapoints.back(), i) datapoints.back()[i] = data[i].first;
	}

	void dump() {
		for (auto &iter : header) fprintf(file, "%s;", iter.c_str());
		fprintf(file, "\n");
		FOR_ALL(datapoints, n) {
			const auto &data = datapoints[n];
			fprintf(file, "%s;", std::to_string(timestamps[n]).c_str());
			for (auto &iter : data) fprintf(file, "%s;", iter.c_str());
			fprintf(file, "\n");
		}
		close();
	}
	void close() {
		if (file)
		{
			fclose(file);
			file = nullptr;
		}
	}
	/* example */
	static void testHL();

};
#endif
int calcJulianDayNumber(const int day, const int month, const int year);
double calcJulianDate(const int day, const int month, const int year, const int hour, const int minute, const int second);
void calcDayFromJulianDayNumber(const int jdn, int& day, int& month, int& year);
void calcDateFromJulianDate(const double jd, int& day, int& month, int& year, int& hour, int& minute, int& second);

template <typename T>
void quick_remove_at(std::vector<T> &v, std::size_t idx)
{
	if (idx < v.size()) {
		v[idx] = std::move(v.back());
		v.pop_back();
	}
}
namespace pawlin {
#if !defined(STM32)
	class Identified {
		mutable std::mutex id_mut;
		static uint64_t id_counter;
	protected:
		const uint64_t id;
	public:
		uint64_t getID() const {
			return id;
		}
		Identified() : id(makeID()) {}
		uint64_t makeID() const {
			std::lock_guard<std::mutex> lock(id_mut);
			auto val = id_counter++;
			return val;
		}
	};

	inline uint64_t microseconds_since_epoch() {
		return std::chrono::duration_cast<std::chrono::microseconds>(
			std::chrono::system_clock::now().time_since_epoch()
			).count();
	}

	inline float dtf(uint64_t ts1_us, uint64_t ts2_us) {
		return float(ts2_us - ts1_us)*(0.001f*0.001f);
	}

	inline string makeUniqueIDFilename(const string &prefix, const string &postfix, int maxtries = 1024, int leadz = 2) {
		string videofilename;
		for (int n = 0; n < maxtries; n++) { // I doubt there can be more than 1024 runs
			videofilename = prefix + std::to_string(makeID())
				+ "_" + leadZ(int2str(n), leadz) + postfix; //leadz
			FILE *file = fopen(videofilename.c_str(), "rb");
			if (file) fclose(file);
			else break;
		}
		return videofilename;
	}
	template <typename T = std::string>
	inline void join(vector<T> &base, const vector<T> &join) {
		base.insert(base.end(), join.begin(), join.end());
	}
#endif

	template<class T, typename Y = float>
	T interpolate(const std::map<Y, T> &keypoints, Y t) {
		if (keypoints.empty()) throw std::runtime_error("interpolate : empty map of keypoints");
		if (keypoints.size() == 1) return keypoints.begin()->second;
		// now we know the size is at least two
		Y ts_start = keypoints.begin()->first;
		Y ts_end = keypoints.rbegin()->first;
		if (t <= ts_start) return keypoints.begin()->second;
		if (t >= ts_end) return keypoints.rbegin()->second;
		auto iter1 = keypoints.upper_bound(t);
		auto iter0 = std::prev(iter1);
		Y t1 = iter1->first;
		Y t0 = iter0->first;
		float tetta = float(double(t - t0) / double(t1 - t0));
		return iter0->second + (iter1->second - iter0->second)*tetta;
	}

	template <typename KeyType, typename ValType>
	class SafeKeyMap {
		std::map<KeyType, ValType> map;
	public:
		size_t size() const {
			return map.size();
		}
		const ValType& operator[](const KeyType &index) const {
			auto iter = map.find(index);
			if (iter == map.end()) throw std::runtime_error("could not find key " + std::to_string(index) + " in safemap");
			return iter->second;
		}
		ValType& operator()(const KeyType &index) { return map[index]; }

		void insertUnique(const KeyType &index, const ValType &val) {
			auto iter = map.find(index);
			if (iter != map.end() && val != iter->second) throw std::runtime_error("non unique value in safemap insertUnique, index " + std::to_string(index));
			map[index] = val;
		}
		std::map<KeyType, ValType> &getMap() { return map; }
		const std::map<KeyType, ValType> &getMap() const { return map; }
		bool has(const KeyType &index) const { return map.count(index) > 0; }

	};

	template <typename T>
	class SafeStringMap {
		std::map<string, T> map;
	public:
		SafeStringMap() {}
		SafeStringMap(const std::map<string, T> &map) : map(map) {}
		const T& operator[](const string &index) const {
			auto iter = map.find(index);
			if (iter == map.end()) throw std::runtime_error("could not find key " + index + " in safemap");
			return iter->second;
		}
		T& operator()(const string &index) { return map[index]; }

		void insertUnique(const string &index, const T &val) {
			auto iter = map.find(index);
			if (iter != map.end() && val != iter->second) throw std::runtime_error("non unique value in safemap insertUnique, index " + index);
			map[index] = val;
		}
		std::map<string, T> &getMap() { return map; }
		const std::map<string, T> &getMap() const { return map; }
		bool has(const string &index) const { return map.count(index) > 0; }

	};

	template<typename Tin, typename Tout>
	class IFunction {
	public:
		virtual ~IFunction() {}
		virtual Tout f(const Tin& x) const = 0;
	};

	inline float deg2rad(float degrees) {
		static float k = M_PIf / 180.0f;
		return degrees*k;
	}
	inline float rad2deg(float rads) {
		static float k = 180.0f / M_PIf;
		return rads*k;
	}

}

bool pwn_stoi(const std::string & str, int & out);
bool pwn_stof(const std::string & str, float & out);
bool pwn_stoll(const std::string & str, long long & out);