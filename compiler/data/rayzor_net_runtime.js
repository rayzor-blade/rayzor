// Rayzor Networking Runtime — WebSocket + fetch for browser
// TCP-like stream → WebSocket (ws:// / wss://)
// DNS → string passthrough

const _socks = new Map();
let _sockN = 1;
const _hosts = new Map();
let _hostN = 1;

const netRuntime = {
  rayzor_socket_new() {
    const h = _sockN++;
    _socks.set(h, { ws: null, recvBuf: [], host: '', port: 0, connected: false });
    return h;
  },
  rayzor_socket_connect(h, hostIp, port) {
    const s = _socks.get(h); if (!s) return;
    const ip = ((hostIp >>> 24) & 0xFF) + '.' + ((hostIp >>> 16) & 0xFF) + '.' +
               ((hostIp >>> 8) & 0xFF) + '.' + (hostIp & 0xFF);
    const host = s.host || ip;
    try {
      const proto = (port === 443 || port === 8443) ? 'wss' : 'ws';
      s.ws = new WebSocket(proto + '://' + host + ':' + port);
      s.ws.binaryType = 'arraybuffer';
      s.ws.onmessage = (e) => {
        const data = e.data instanceof ArrayBuffer ? new Uint8Array(e.data) : new TextEncoder().encode(e.data);
        s.recvBuf.push(data);
      };
      s.ws.onopen = () => { s.connected = true; };
      s.ws.onclose = () => { s.connected = false; };
    } catch(e) { console.warn('[rayzor:net]', e); }
  },
  rayzor_socket_bind() {},
  rayzor_socket_listen() {},
  rayzor_socket_accept() { return 0; },
  rayzor_socket_close(h) { const s = _socks.get(h); if (s?.ws) s.ws.close(); _socks.delete(h); },
  rayzor_socket_read(h) {
    const s = _socks.get(h); if (!s || !s.recvBuf.length) return 0;
    return writeString(new TextDecoder().decode(s.recvBuf.shift()));
  },
  rayzor_socket_write(h, strPtr) {
    const s = _socks.get(h); if (!s?.ws || s.ws.readyState !== 1) return;
    try { s.ws.send(readString(strPtr)); } catch {}
  },
  rayzor_socket_shutdown(h) { const s = _socks.get(h); if (s?.ws) s.ws.close(); },
  rayzor_socket_set_blocking() {},
  rayzor_socket_set_timeout() {},
  rayzor_socket_set_fast_send() {},
  rayzor_socket_wait_for_read() {},
  rayzor_socket_select() { return 0; },
  rayzor_socket_peer() { return 0; },
  rayzor_socket_host_info() { return 0; },
  rayzor_socket_get_input(h) { return h; },
  rayzor_socket_get_output(h) { return h; },
  rayzor_socket_read_byte(h) {
    const s = _socks.get(h); if (!s || !s.recvBuf.length) return -1;
    const buf = s.recvBuf[0]; const b = buf[0];
    if (buf.length === 1) s.recvBuf.shift(); else s.recvBuf[0] = buf.subarray(1);
    return b;
  },
  rayzor_socket_read_bytes(h, bytesPtr, pos, len) {
    const s = _socks.get(h); if (!s || !memory) return 0;
    let read = 0; const dst = new Uint8Array(memory.buffer);
    while (read < len && s.recvBuf.length > 0) {
      const buf = s.recvBuf[0]; const take = Math.min(buf.length, len - read);
      dst.set(buf.subarray(0, take), bytesPtr + pos + read); read += take;
      if (take >= buf.length) s.recvBuf.shift(); else s.recvBuf[0] = buf.subarray(take);
    }
    return read;
  },
  rayzor_socket_write_byte(h, b) {
    const s = _socks.get(h); if (s?.ws?.readyState === 1) s.ws.send(new Uint8Array([b]));
  },
  rayzor_socket_write_bytes(h, bytesPtr, pos, len) {
    const s = _socks.get(h); if (!s?.ws || s.ws.readyState !== 1 || !memory) return 0;
    s.ws.send(new Uint8Array(memory.buffer, bytesPtr + pos, len)); return len;
  },
  rayzor_socket_write_string(h, strPtr) { netRuntime.rayzor_socket_write(h, strPtr); return 1; },
  rayzor_socket_flush() {},

  // Host (DNS) — string passthrough
  rayzor_host_new(namePtr) {
    const name = readString(namePtr); const h = _hostN++;
    _hosts.set(h, { name, ip: 0x7f000001 }); return h;
  },
  rayzor_host_get_ip(h) { return _hosts.get(h)?.ip ?? 0; },
  rayzor_host_to_string(h) {
    const host = _hosts.get(h); if (!host) return 0;
    const ip = host.ip;
    return writeString(((ip>>>24)&0xFF)+'.'+((ip>>>16)&0xFF)+'.'+((ip>>>8)&0xFF)+'.'+(ip&0xFF));
  },
  rayzor_host_reverse(h) { return netRuntime.rayzor_host_to_string(h); },
  rayzor_host_localhost() { return writeString('localhost'); },

  // SSL — delegates to WebSocket (wss://) which handles TLS natively
  rayzor_ssl_socket_new() { return netRuntime.rayzor_socket_new(); },
  rayzor_ssl_socket_connect(h, ip, port) { netRuntime.rayzor_socket_connect(h, ip, port || 443); },
  rayzor_ssl_socket_handshake() { return 0; },
  rayzor_ssl_socket_set_hostname(h, namePtr) { const s = _socks.get(h); if (s) s.host = readString(namePtr); },
  rayzor_ssl_socket_set_ca() {}, rayzor_ssl_socket_set_certificate() {},
  rayzor_ssl_socket_peer_certificate() { return 0; },
  rayzor_ssl_socket_read(h) { return netRuntime.rayzor_socket_read(h); },
  rayzor_ssl_socket_write(h, s) { netRuntime.rayzor_socket_write(h, s); },
  rayzor_ssl_socket_close(h) { netRuntime.rayzor_socket_close(h); },
  rayzor_ssl_socket_set_blocking() {}, rayzor_ssl_socket_set_timeout() {},
  rayzor_ssl_socket_get_input(h) { return h; }, rayzor_ssl_socket_get_output(h) { return h; },
  rayzor_ssl_socket_shutdown(h) { netRuntime.rayzor_socket_shutdown(h); },
  rayzor_ssl_socket_set_fast_send() {},
  rayzor_ssl_socket_read_byte(h) { return netRuntime.rayzor_socket_read_byte(h); },
  rayzor_ssl_socket_read_bytes(h, b, p, l) { return netRuntime.rayzor_socket_read_bytes(h, b, p, l); },
  rayzor_ssl_socket_write_byte(h, b) { netRuntime.rayzor_socket_write_byte(h, b); },
  rayzor_ssl_socket_write_bytes(h, b, p, l) { return netRuntime.rayzor_socket_write_bytes(h, b, p, l); },
  rayzor_ssl_socket_write_string(h, s) { return netRuntime.rayzor_socket_write_string(h, s); },
  rayzor_ssl_socket_flush() {},
  rayzor_ssl_cert_load_file() { return 0; }, rayzor_ssl_cert_load_path() { return 0; },
  rayzor_ssl_cert_from_string() { return 0; }, rayzor_ssl_cert_load_defaults() { return 0; },
  rayzor_ssl_cert_common_name() { return 0; }, rayzor_ssl_cert_alt_names() { return 0; },
  rayzor_ssl_cert_not_before() { return 0; }, rayzor_ssl_cert_not_after() { return 0; },
  rayzor_ssl_cert_subject() { return 0; }, rayzor_ssl_cert_issuer() { return 0; },
  rayzor_ssl_cert_next() { return 0; }, rayzor_ssl_cert_add() { return 0; },
  rayzor_ssl_cert_add_der() { return 0; }, rayzor_ssl_key_load_file() { return 0; },
  rayzor_ssl_key_read_pem() { return 0; }, rayzor_ssl_key_read_der() { return 0; },
  rayzor_ssl_digest_make() { return 0; }, rayzor_ssl_digest_sign() { return 0; },
  rayzor_ssl_digest_verify() { return 0; },
};
