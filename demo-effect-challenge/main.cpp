#include "app.h"

#include <Windows.h>
#include <tchar.h>

namespace
{
	bool mouseButtonDown[2] = { false, false };
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
	switch (uMsg)
	{
	case WM_LBUTTONDOWN:
		mouseButtonDown[0] = true;
		break;
	case WM_RBUTTONDOWN:
		mouseButtonDown[1] = true;
		break;
	case WM_LBUTTONUP:
		mouseButtonDown[0] = false;
		break;
	case WM_RBUTTONUP:
		mouseButtonDown[1] = false;
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		return 0;
	}
	return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

int main() {
	const int width = 256;
	const int height = 256;

	const TCHAR* wndClassName = _T("DemoEffectWindowClass");
	WNDCLASS wndClass = { 0 };
	wndClass.style = 0;
	wndClass.lpfnWndProc = WindowProc;
	wndClass.hInstance = GetModuleHandle(NULL);
	wndClass.lpszClassName = wndClassName;

	RegisterClass(&wndClass);
	const auto hwnd = CreateWindow(wndClassName, _T("Demo Effect Challenge"), WS_OVERLAPPED, CW_USEDEFAULT, 0, 2 * width, 2 * height, 
		GetDesktopWindow(), NULL, GetModuleHandle(NULL), NULL);
	ShowWindow(hwnd, SW_SHOWNORMAL);

	auto app = CreateApp(hwnd, width, height);

	MSG msg = { 0 };
	do {
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		} else {
			RECT windowRect;
			GetWindowRect(hwnd, &windowRect);
			POINT cursorPos;
			GetCursorPos(&cursorPos);
			int mouseX = cursorPos.x - windowRect.left;
			int mouseY = cursorPos.y - windowRect.top;
			app->Update(mouseButtonDown, mouseX, mouseY);
		}
	} while (msg.message != WM_QUIT);

	return 0;
}
