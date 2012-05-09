#pragma once

#include <memory>

class IApp {
public:
	virtual ~IApp() {}
	virtual void Update(bool mouseButtons[2], int mouseX, int mouseY) = 0;
};

std::unique_ptr<IApp> CreateApp(void* hwnd, int width, int height);
