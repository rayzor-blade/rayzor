//! The server's stdio transport.
//!
//! The protocol owns stdout, but compiler code can print there; one stray line
//! corrupts the stream and the editor drops the session. On Unix the transport
//! writes to a private copy of stdout and fd 1 is pointed at stderr, so such
//! output lands in the server log. The reader queues every message as it
//! arrives, so the server sees a burst of edits at once and can coalesce it.

use crossbeam_channel::unbounded;
use lsp_server::{Connection, Message};
use std::io::{self, BufReader, Write};
use std::thread::{self, JoinHandle};

pub type IoThread = JoinHandle<io::Result<()>>;

pub fn stdio() -> (Connection, [IoThread; 2]) {
    let mut out = protocol_output();
    let (writer_sender, writer_receiver) = unbounded::<Message>();
    let writer = thread::Builder::new()
        .name("lsp-writer".into())
        .spawn(move || {
            for message in writer_receiver {
                message.write(&mut out)?;
                out.flush()?;
            }
            Ok(())
        })
        .expect("spawn lsp writer");

    let (reader_sender, reader_receiver) = unbounded::<Message>();
    let reader = thread::Builder::new()
        .name("lsp-reader".into())
        .spawn(move || {
            let mut input = BufReader::new(io::stdin());
            while let Some(message) = Message::read(&mut input)? {
                let exit = matches!(&message, Message::Notification(n) if n.method == "exit");
                if reader_sender.send(message).is_err() || exit {
                    break;
                }
            }
            Ok(())
        })
        .expect("spawn lsp reader");

    let connection = Connection {
        sender: writer_sender,
        receiver: reader_receiver,
    };
    (connection, [writer, reader])
}

#[cfg(unix)]
fn protocol_output() -> Box<dyn Write + Send> {
    use std::os::fd::FromRawFd;
    // SAFETY: dup returns a fresh descriptor this function alone owns, and
    // dup2 only retargets fd 1, which nothing else in the process holds as a
    // `File`.
    unsafe {
        let private = libc::dup(libc::STDOUT_FILENO);
        if private < 0 {
            return Box::new(io::stdout());
        }
        libc::dup2(libc::STDERR_FILENO, libc::STDOUT_FILENO);
        Box::new(std::fs::File::from_raw_fd(private))
    }
}

#[cfg(not(unix))]
fn protocol_output() -> Box<dyn Write + Send> {
    Box::new(io::stdout())
}
