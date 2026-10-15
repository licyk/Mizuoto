import { defineStore } from 'pinia';
import { ref } from 'vue';

const KEY = 'mizuoto:token';
export const TOKEN_COOKIE = 'mizuoto_token';

function readToken(): string | null {
  try {
    return localStorage.getItem(KEY);
  } catch {
    return null;
  }
}

function writeCookie(token: string | null) {
  // The cookie lets <img> previews and the socket handshake carry the token; the API also accepts the header.
  document.cookie = token
    ? `${TOKEN_COOKIE}=${encodeURIComponent(token)}; path=/; SameSite=Strict`
    : `${TOKEN_COOKIE}=; path=/; max-age=0; SameSite=Strict`;
}

/** Access token for servers that require one (non-loopback hosts). */
export const useAuthStore = defineStore('auth', () => {
  const token = ref<string | null>(readToken());
  const needed = ref(false);
  if (token.value) writeCookie(token.value);

  function setToken(value: string | null) {
    token.value = value;
    try {
      if (value) localStorage.setItem(KEY, value);
      else localStorage.removeItem(KEY);
    } catch {
      /* private mode: keep it in memory */
    }
    writeCookie(value);
    needed.value = false;
  }
  return { token, needed, setToken };
});
