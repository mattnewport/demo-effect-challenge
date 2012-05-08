#include "app.h"

#include <Windows.h>
#include <tchar.h>

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
	switch (uMsg)
	{
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
			app->Update();
		}
	} while (msg.message != WM_QUIT);

	return 0;
}
