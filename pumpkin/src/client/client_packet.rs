use std::{rc::Rc, sync::Mutex};

use crate::protocol::{
    client::{
        config::CFinishConfig,
        login::{CEncryptionRequest, CLoginSuccess},
        status::{CPingResponse, CStatusResponse},
    },
    server::{
        config::{SAcknowledgeFinishConfig, SClientInformation},
        handshake::SHandShake,
        login::{SEncryptionResponse, SLoginAcknowledged, SLoginPluginResponse, SLoginStart},
        status::{SPingRequest, SStatusRequest},
    },
    ConnectionState,
};

use super::{Client, PlayerConfig};

pub trait ClientPacketProcessor {
    // Handshake
    fn handle_handshake(&mut self, handshake: SHandShake);
    // Status
    fn handle_status_request(&mut self, status_request: SStatusRequest);
    fn handle_ping_request(&mut self, ping_request: SPingRequest);
    // Login
    fn handle_login_start(&mut self, login_start: SLoginStart);
    fn handle_encryption_response(&mut self, encryption_response: SEncryptionResponse);
    fn handle_plugin_response(&mut self, plugin_response: SLoginPluginResponse);
    fn handle_login_acknowledged(&mut self, login_acknowledged: SLoginAcknowledged);
    // Config
    fn handle_client_information(&mut self, client_information: SClientInformation);
    fn handle_config_acknowledged(&mut self, config_acknowledged: SAcknowledgeFinishConfig);
}

impl ClientPacketProcessor for Client {
    fn handle_handshake(&mut self, handshake: SHandShake) {
        // TODO set protocol version and check protocol version
        self.connection_state = handshake.next_state;
        dbg!("handshake");
    }

    fn handle_status_request(&mut self, _status_request: SStatusRequest) {
        dbg!("sending status");

        dbg!("test first");
        
        let guard = self.server.try_lock().unwrap();
        dbg!("test");

        let response = serde_json::to_string(&guard.status_response).unwrap();
        drop(guard);

        self.send_packet(CStatusResponse::new(response));
    }

    fn handle_ping_request(&mut self, ping_request: SPingRequest) {
        dbg!("ping");
        self.send_packet(CPingResponse::new(ping_request.payload));
        self.close();
    }

    fn handle_login_start(&mut self, login_start: SLoginStart) {
        dbg!("login start");
        self.name = Some(login_start.name);
        self.uuid = Some(login_start.uuid);
        let verify_token: [u8; 4] = rand::random();
        let public_key_der = self
            .server
            .to_owned()
            .lock()
            .unwrap()
            .public_key_der
            .clone(); // todo do not clone
        let packet = CEncryptionRequest::new(
            "".into(),
            public_key_der.len() as i32,
            &public_key_der,
            verify_token.len() as i32,
            &verify_token,
            false, // TODO
        );
        self.send_packet(packet);
    }

    fn handle_encryption_response(&mut self, encryption_response: SEncryptionResponse) {
        dbg!("encryption response");
        // should be impossible
        if self.uuid.is_none() || self.name.is_none() {
            self.kick("UUID or Name is none".into());
            return;
        }
        self.enable_encryption(encryption_response.shared_secret)
            .unwrap();

        let packet = CLoginSuccess::new(self.uuid.unwrap(), self.name.clone().unwrap(), 0, false);
        self.send_packet(packet);
    }

    fn handle_plugin_response(&mut self, plugin_response: SLoginPluginResponse) {}

    fn handle_login_acknowledged(&mut self, login_acknowledged: SLoginAcknowledged) {
        self.connection_state = ConnectionState::Config;
        dbg!("login achnowlaged");
    }
    fn handle_client_information(&mut self, client_information: SClientInformation) {
        self.config = Some(PlayerConfig {
            locale: client_information.locale,
            view_distance: client_information.view_distance,
            chat_mode: client_information.chat_mode,
            chat_colors: client_information.chat_colors,
            skin_parts: client_information.skin_parts,
            main_hand: client_information.main_hand,
            text_filtering: client_information.text_filtering,
            server_listing: client_information.server_listing,
        });
        // We are done with configuring
        self.send_packet(CFinishConfig::new());
    }

    fn handle_config_acknowledged(&mut self, config_acknowledged: SAcknowledgeFinishConfig) {
        dbg!("config acknowledged");
        self.connection_state = ConnectionState::Play;
        // generate a player
        self.server.lock().unwrap().spawn_player(&self.token);
    }
}