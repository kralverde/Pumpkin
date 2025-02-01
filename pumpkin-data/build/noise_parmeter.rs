use std::collections::HashMap;

use proc_macro2::TokenStream;
use quote::quote;
use serde::Deserialize;

use crate::ident;

#[derive(Deserialize)]
pub struct DoublePerlinNoiseParameters {
    #[serde(rename = "firstOctave")]
    first_octave: i32,
    amplitudes: Vec<f64>,
}

pub(crate) fn build() -> TokenStream {
    println!("cargo:rerun-if-changed=../assets/noise_parameters.json");

    let json: HashMap<String, DoublePerlinNoiseParameters> =
        serde_json::from_str(include_str!("../../assets/noise_parameters.json"))
            .expect("Failed to parse noise_parameters.json");
    let mut variants = TokenStream::new();
    let mut match_variants = TokenStream::new();

    for (name, paremter) in json.iter() {
        let raw_name = format!("minecraft:{name}");
        let simple_id = name;
        let name = ident(name.to_uppercase());
        let first_octave = paremter.first_octave;
        let amplitudes = &paremter.amplitudes;
        variants.extend([quote! {
            pub const #name: DoublePerlinNoiseParameters = DoublePerlinNoiseParameters::new(#first_octave, &[#(#amplitudes),*], #raw_name);
        }]);
        match_variants.extend([quote! {
            #simple_id => &#name,
        }]);
    }

    quote! {
        pub struct DoublePerlinNoiseParameters {
            pub first_octave: i32,
            pub amplitudes: &'static [f64],
            id: &'static str,
        }

        impl DoublePerlinNoiseParameters {
            pub const fn new(first_octave: i32, amplitudes: &'static [f64], id: &'static str) -> Self {
                Self {
                    first_octave,
                    amplitudes,
                    id
                }
            }

            pub const fn id(&self) -> &'static str {
                self.id
            }

            pub fn id_to_parameters(id: &str) -> Option<&DoublePerlinNoiseParameters> {
                Some(match id {
                    #match_variants
                    _ => return None,
                })
            }
        }

        #variants
    }
}
