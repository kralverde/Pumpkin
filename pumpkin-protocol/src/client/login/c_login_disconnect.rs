use pumpkin_macros::client_packet;
use serde::Serialize;

#[derive(Serialize)]
#[client_packet("login:login_disconnect")]
pub struct CLoginDisconnect<'a> {
    json_reason: &'a str,
}

impl<'a> CLoginDisconnect<'a> {
    // input json!
    pub fn new(json_reason: &'a str) -> Self {
        Self { json_reason }
    }
}
