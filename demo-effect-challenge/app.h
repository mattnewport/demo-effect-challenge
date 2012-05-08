#pragma once

#include <memory>

class IApp {
public:
	virtual ~IApp() {}
	virtual void Update() = 0;
};

std::unique_ptr<IApp> CreateApp(void* hwnd, int width, int height);
