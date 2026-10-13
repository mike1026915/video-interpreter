// 點工具列圖示：打開側邊欄。這次點擊同時讓擴充功能取得擷取該分頁聲音的權限。
chrome.action.onClicked.addListener((tab) => {
  chrome.sidePanel.open({ tabId: tab.id });
});
