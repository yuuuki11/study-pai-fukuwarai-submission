// Persistent Fortress transport client. Keeps policy ticks independent of CLI startup.
#include <ignition/transport/Node.hh>
#include <ignition/msgs/world_control.pb.h>
#include <ignition/msgs/boolean.pb.h>
#include <iostream>
#include <string>
int main() {
  ignition::transport::Node node;
  std::string command;
  unsigned int value;
  while (std::cin >> command >> value) {
    ignition::msgs::WorldControl request;
    ignition::msgs::Boolean reply;
    bool result = false;
    if (command == "pause") request.set_pause(value != 0);
    else if (command == "step") request.set_multi_step(value);
    else if (command == "paused_step") { request.set_pause(true); request.set_multi_step(value); }
    else { std::cout << "ERROR unknown command" << std::endl; continue; }
    bool executed = node.Request("/world/default/control", request, 10000, reply, result);
    std::cout << ((executed && result && reply.data()) ? "OK" : "ERROR request failed") << std::endl;
  }
}
