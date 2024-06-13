// File: PwnParams.h / cpp
// Purpose: class PwnParams use in any algorithms
// Author: Alex Dolgopolov
// Date: 29-07-2010
// Version 1.0
// (C) PAWLIN TECHNOLOGIES LTD. ALL RIGHTS RESERVED

#pragma once

#include <vector>
#include <map>
//#include <xtree>
#include <string>
#include <algorithm>
#include <cctype>       // std::toupper tolower
#include <iostream>

using std::vector;
using std::map;
using std::string;

//#define _VS_2012

union Multitype {
		bool b;
		unsigned u;
		int i;
		float f;
};

class PwnParams:public  map<string, Multitype>{
public:
	PwnParams(){}
	Multitype & operator[](const string& keyval)
	{	
#ifdef _VS_2012
		string _Keyval = keyval;
		// explicit cast needed to resolve ambiguity
		std::transform(_Keyval.begin(), _Keyval.end(), _Keyval.begin(),(int(*)(int)) std::tolower);
		// find element matching _Keyval or insert with default mapped
		iterator _Where = this->lower_bound(_Keyval);
		if (_Where == this->end()
			|| this->_Getcomp()(_Keyval, this->_Key(_Where._Mynode()))){
			_Where = this->insert(_Where,
			value_type(_Keyval, mapped_type()));
		}
		return ((*_Where).second);
#else
		/*string _Keyval = keyval;
		// explicit cast needed to resolve ambiguity
		std::transform(_Keyval.begin(), _Keyval.end(), _Keyval.begin(),(int(*)(int)) std::tolower);
		// find element matching _Keyval or insert with default mapped
		iterator _Where = this->lower_bound(_Keyval);
		if (_Where == this->end()
			|| this->comp(_Keyval, this->_Key(_Where._Mynode()))){
			_Where = this->insert(_Where,
			value_type(_Keyval, mapped_type()));
		}
		return ((*_Where).second);*/
	    string _Keyval = keyval;
		// explicit cast needed to resolve ambiguity
		std::transform(_Keyval.begin(), _Keyval.end(), _Keyval.begin(),(int(*)(int)) std::tolower);
		// find element matching _Keyval or insert with default mapped
		iterator _Where = this->lower_bound(_Keyval);
		if (_Where == this->end()
			|| _Keyval.compare((*_Where).first)){
			_Where = this->insert(_Where,
			value_type(_Keyval, mapped_type()));
		}
		return ((*_Where).second);


#endif
	}
	const Multitype  operator[](const string& keyval) const
	{	

#ifdef _VS_2012
		string _Keyval = keyval;
		std::transform(_Keyval.begin(), _Keyval.end(), _Keyval.begin(),(int(*)(int)) std::tolower);

		// find element matching _Keyval or insert with default mapped
		const_iterator _Where = this->lower_bound(_Keyval);
		if (_Where == this->end()
			|| this->_Getcomp()(_Keyval, this->_Key(_Where._Mynode()))){

				return Multitype();
				//throw("Algoparams:Invalid argument");
		}
		return ((*_Where).second);
#else
		/*string _Keyval = keyval;
		std::transform(_Keyval.begin(), _Keyval.end(), _Keyval.begin(),(int(*)(int)) std::tolower);

		// find element matching _Keyval or insert with default mapped
		const_iterator _Where = this->lower_bound(_Keyval);
		if (_Where == this->end()
			|| this->comp(_Keyval, this->_Key(_Where._Mynode()))){

				return Multitype();
				//throw("Algoparams:Invalid argument");
		}
		return ((*_Where).second);*/
		
		string _Keyval = keyval;
		std::transform(_Keyval.begin(), _Keyval.end(), _Keyval.begin(),(int(*)(int)) std::tolower);

		// find element matching _Keyval or insert with default mapped
		const_iterator _Where = this->lower_bound(_Keyval);
		if (_Where == this->end()
			|| _Keyval.compare((*_Where).first)){

				return Multitype();
				//throw("Algoparams:Invalid argument");
		}
		return ((*_Where).second);

#endif
	}
	bool getParam(const string& keyval, Multitype & param){
#ifdef _VS_2012
		string _Keyval = keyval;
		std::transform(_Keyval.begin(), _Keyval.end(), _Keyval.begin(),(int(*)(int)) std::tolower);
		const_iterator _Where = this->lower_bound(_Keyval);
		if (_Where == this->end()
			|| this->_Getcomp()(_Keyval, this->_Key(_Where._Mynode()))){
			param = Multitype();
			return false;
		}
		param = ((*_Where).second);
		return true;
#else
		/*string _Keyval = keyval;
		std::transform(_Keyval.begin(), _Keyval.end(), _Keyval.begin(),(int(*)(int)) std::tolower);
		const_iterator _Where = this->lower_bound(_Keyval);
		if (_Where == this->end()
			|| this->comp(_Keyval, this->_Key(_Where._Mynode()))){
			param = Multitype();
			return false;

		}
		param = ((*_Where).second);
		return true;*/
		string _Keyval = keyval;
		std::transform(_Keyval.begin(), _Keyval.end(), _Keyval.begin(),(int(*)(int)) std::tolower);
		const_iterator _Where = this->lower_bound(_Keyval);
		if (_Where == this->end()
			|| _Keyval.compare((*_Where).first)){
			param = Multitype();
			return false;
		}
		param = ((*_Where).second);
		return true;

#endif
	}
	~PwnParams(){}
};