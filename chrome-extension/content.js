const SELECTORS = [
  "ytd-comments ytd-comment-thread-renderer #content-text",
  "ytd-comment-thread-renderer #content-text",
  "ytd-comment-renderer #content-text",
];

function extractComments() {
  const seen = new Set();
  const comments = [];
  for (const selector of SELECTORS) {
    document.querySelectorAll(selector).forEach((el) => {
      const text = el.innerText.trim();
      if (text && !seen.has(text)) {
        seen.add(text);
        comments.push(text);
      }
    });
    if (comments.length > 0) break;
  }
  return comments;
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function loadMoreComments(rounds) {
  const startY = window.scrollY;
  const section = document.querySelector("ytd-comments");
  if (section) section.scrollIntoView();
  await wait(1200);
  for (let i = 0; i < rounds; i++) {
    window.scrollTo(0, document.documentElement.scrollHeight);
    await wait(1200);
  }
  window.scrollTo(0, startY);
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "getComments") {
    loadMoreComments(message.scrollRounds || 0).then(() => {
      const comments = extractComments().slice(0, message.limit || 1000);
      sendResponse({ comments });
    });
    return true;
  }
});
