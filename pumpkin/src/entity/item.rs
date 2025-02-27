use std::sync::{Arc, atomic::AtomicU32};

use async_trait::async_trait;
use pumpkin_data::item::Item;
use pumpkin_protocol::{
    client::play::{CTakeItemEntity, MetaDataType, Metadata},
    codec::slot::Slot,
};
use pumpkin_world::item::ItemStack;
use tokio::sync::Mutex;

use super::{Entity, EntityBase, living::LivingEntity, player::Player};

pub struct ItemEntity {
    entity: Entity,
    item: Item,
    item_age: AtomicU32,
    // These cannot be atomic values because we mutate their state based on what they are; we run
    // into the ABA problem
    item_count: Mutex<u32>,
    pickup_delay: Mutex<u8>,
}

impl ItemEntity {
    pub fn new(entity: Entity, item_id: u16, count: u32) -> Self {
        Self {
            entity,
            item: Item::from_id(item_id).expect("We passed a bad item id into ItemEntity"),
            item_age: AtomicU32::new(0),
            item_count: Mutex::new(count),
            pickup_delay: Mutex::new(10), // Vanilla pickup delay is 10 ticks
        }
    }
    pub async fn send_meta_packet(&self) {
        let slot = Slot::new(self.item.id, *self.item_count.lock().await);
        self.entity
            .send_meta_data(Metadata::new(8, MetaDataType::ItemStack, &slot))
            .await;
    }
}

#[async_trait]
impl EntityBase for ItemEntity {
    async fn tick(&self) {
        {
            let mut delay = self.pickup_delay.lock().await;
            *delay = delay.saturating_sub(1);
        }

        let age = self
            .item_age
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if age >= 6000 {
            self.entity.remove().await;
        }
    }
    async fn on_player_collision(&self, player: Arc<Player>) {
        let can_pickup = {
            let delay = self.pickup_delay.lock().await;
            *delay == 0
        };

        if can_pickup {
            let mut inv = player.inventory.lock().await;
            let mut total_pick_up = 0;
            let mut remove_entity = false;
            let mut slot_updates = Vec::new();

            {
                let mut stack_size = self.item_count.lock().await;
                let max_stack = self.item.components.max_stack_size;
                while *stack_size > 0 {
                    if let Some(slot) = inv.collect_item_slot(self.item.id) {
                        // Fill the inventory while there are items in the stack and space in the inventory
                        if let Some(existing_stack) = inv
                            .get_slot(slot)
                            .expect("collect item slot returned an invalid slot")
                        {
                            // We have the item in this stack already

                            // This is bounded to u8::MAX
                            let amount_to_fill = (max_stack - existing_stack.item_count) as u32;
                            // This is also bounded to u8::MAX since amount_to_fill is max u8::MAX
                            let amount_to_add = amount_to_fill.min(*stack_size);
                            // Therefore this is safe
                            existing_stack.item_count += amount_to_add as u8;
                            total_pick_up += amount_to_add;
                            *stack_size -= amount_to_add;

                            slot_updates.push((slot, existing_stack.clone()));
                        } else {
                            // A new stack

                            // This is bounded to u8::MAX
                            let amount_to_fill = max_stack as u32;
                            // This is also bounded to u8::MAX since amount_to_fill is max u8::MAX
                            let amount_to_add = amount_to_fill.min(*stack_size);
                            total_pick_up += amount_to_add;
                            *stack_size -= amount_to_add;

                            // Therefore this is safe
                            let item_stack = ItemStack::new(amount_to_add as u8, self.item.clone());
                            slot_updates.push((slot, item_stack));
                        }
                    } else {
                        // We can't pick anything else up
                        break;
                    }
                }
                if *stack_size == 0 {
                    remove_entity = true;
                }
            }

            if total_pick_up > 0 {
                player
                    .client
                    .send_packet(&CTakeItemEntity::new(
                        self.entity.entity_id.into(),
                        player.entity_id().into(),
                        total_pick_up.into(),
                    ))
                    .await;
            }

            // TODO: Can we batch slot updates?
            for (slot, stack) in slot_updates {
                player.update_single_slot(&mut inv, slot, stack).await;
            }

            if remove_entity {
                self.entity.remove().await;
            } else {
                // Update entity
                self.send_meta_packet().await;
            }
        }
    }

    fn get_entity(&self) -> &Entity {
        &self.entity
    }

    fn get_living_entity(&self) -> Option<&LivingEntity> {
        None
    }
}
