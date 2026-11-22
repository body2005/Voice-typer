@echo off
:: VoiceTyper - Windows Defender & SmartScreen Helper
title VoiceTyper - السماح بتشغيل البرنامج في حماية ويندوز
chcp 65001 >nul
cls

echo ========================================================
echo        VoiceTyper - السماح بالتشغيل في حماية ويندوز
echo ========================================================
echo.
echo جاري فك حظر ملفات البرنامج (Unblock Files)...

powershell -NoProfile -Command "Get-ChildItem -Path '%~dp0' -Recurse | Unblock-File -ErrorAction SilentlyContinue"

echo [✓] تم فك حظر الملفات بنجاح من علامة التنزيل (SmartScreen Mark of the Web).
echo.
echo إذا كان Windows Defender يوقف البرنامج، يمكنك إضافة هذا المجلد كاستثناء آمن:
echo هل تريد إضافة مجلد VoiceTyper كاستثناء في Windows Defender تلقائياً؟ (Y/N)
set /p user_choice="اختر (Y/N): "

if /i "%user_choice%"=="Y" (
    echo جاري طلب صلاحية المسؤول لإضافة الاستثناء...
    powershell -Command "Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile -Command Add-MpPreference -ExclusionPath ''%~dp0'''"
    echo [✓] تم إرسال أمر إضافة الاستثناء بنجاح.
)

echo.
echo ========================================================
echo يمكنك الآن تشغيل VoiceTyper-Standalone.exe مباشرة بكل سهولة!
echo ========================================================
pause
