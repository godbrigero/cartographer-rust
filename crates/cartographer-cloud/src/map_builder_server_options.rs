//! Port of `cartographer/cloud/map_builder_server_options` (.h, .cc).

use cartographer::common::configuration_file_resolver::ConfigurationFileResolver;
use cartographer::common::lua_parameter_dictionary::{FileResolver, LuaParameterDictionary};
use cartographer::mapping::map_builder_interface::create_map_builder_options;

use crate::proto;

/// Reads `MapBuilderServerOptions` from `lua_parameter_dictionary`.
pub fn create_map_builder_server_options(
    lua_parameter_dictionary: &LuaParameterDictionary,
) -> proto::MapBuilderServerOptions {
    let server_address = lua_parameter_dictionary.get_string("server_address");
    let num_grpc_threads = lua_parameter_dictionary.get_int("num_grpc_threads");
    let num_event_threads = lua_parameter_dictionary.get_int("num_event_threads");
    let map_builder_options =
        create_map_builder_options(&lua_parameter_dictionary.get_dictionary("map_builder"));
    let uplink_server_address = lua_parameter_dictionary.get_string("uplink_server_address");
    let upload_batch_size = lua_parameter_dictionary.get_int("upload_batch_size");
    let enable_ssl_encryption = lua_parameter_dictionary.get_bool("enable_ssl_encryption");
    let enable_google_auth = lua_parameter_dictionary.get_bool("enable_google_auth");
    proto::MapBuilderServerOptions {
        server_address,
        num_grpc_threads,
        num_event_threads,
        map_builder_options: Some(map_builder_options),
        uplink_server_address,
        upload_batch_size,
        enable_ssl_encryption,
        enable_google_auth,
    }
}

/// Loads `configuration_basename` from `configuration_directory` (falling
/// back to the Cartographer configuration files) and reads the options.
pub fn load_map_builder_server_options(
    configuration_directory: &str,
    configuration_basename: &str,
) -> proto::MapBuilderServerOptions {
    let file_resolver = ConfigurationFileResolver::new([configuration_directory]);
    let code = file_resolver.get_file_content_or_die(configuration_basename);
    let lua_parameter_dictionary = LuaParameterDictionary::new(&code, Box::new(file_resolver));
    create_map_builder_server_options(&lua_parameter_dictionary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cartographer::mapping::internal::testing::test_helpers::resolve_lua_parameters;

    #[test]
    fn reads_the_default_configuration() {
        let options = create_map_builder_server_options(&resolve_lua_parameters(
            r#"include "map_builder_server.lua"
            MAP_BUILDER.use_trajectory_builder_2d = true
            return MAP_BUILDER_SERVER"#,
        ));
        assert_eq!(options.server_address, "0.0.0.0:50051");
        assert_eq!(options.num_grpc_threads, 4);
        assert_eq!(options.num_event_threads, 4);
        assert_eq!(options.uplink_server_address, "");
        assert_eq!(options.upload_batch_size, 100);
        assert!(!options.enable_ssl_encryption);
        assert!(!options.enable_google_auth);
        let map_builder_options = options.map_builder_options.unwrap();
        assert!(map_builder_options.use_trajectory_builder_2d);
        assert!(map_builder_options.collate_by_trajectory);
    }

    #[test]
    fn loads_from_a_configuration_directory() {
        let directory = std::env::temp_dir().join(format!(
            "cartographer_cloud_options_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("my_server.lua"),
            r#"include "map_builder_server.lua"
            MAP_BUILDER.use_trajectory_builder_3d = true
            MAP_BUILDER_SERVER.server_address = "127.0.0.1:0"
            MAP_BUILDER_SERVER.upload_batch_size = 7
            return MAP_BUILDER_SERVER"#,
        )
        .unwrap();
        let options = load_map_builder_server_options(directory.to_str().unwrap(), "my_server.lua");
        std::fs::remove_dir_all(&directory).unwrap();
        assert_eq!(options.server_address, "127.0.0.1:0");
        assert_eq!(options.upload_batch_size, 7);
        assert!(
            options
                .map_builder_options
                .unwrap()
                .use_trajectory_builder_3d
        );
    }
}
