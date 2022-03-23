#include <iostream>
#include "dshark_common.h"
#include "dshark_message.h"
#include "dshark_client.h"
#include "dshark_server.h"
#include <asio.hpp>
#include <fstream>
#include <iterator>

enum class dSHARKMessageType : uint32_t
{
	EvaluateBinary,
	ServerAccept,
	ServerDeny,
	ServerPing,
	MessageAll,
	ServerMessage,
};

class CustomClient : public dshark::client_interface<dSHARKMessageType>
{
public:
	void PingServer()
	{
		dshark::message<dSHARKMessageType> msg;
		msg.header.id = dSHARKMessageType::ServerPing;

		// Caution with this...
		std::chrono::system_clock::time_point timeNow = std::chrono::system_clock::now();

		msg << timeNow;
		Send(msg);
	}

	void MessageAll()
	{
		dshark::message<dSHARKMessageType> msg;
		msg.header.id = dSHARKMessageType::MessageAll;
		Send(msg);
	}

	void EvaluateBinary(std::string filepath)
	{
		std::cout << "Loading Binary" << std::endl;
		std::ifstream input(filepath, std::ios::binary);
		std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});
		dshark::message<dSHARKMessageType> msg;
		msg.header.id = dSHARKMessageType::EvaluateBinary;
		for (int i = 0; i < buffer.size(); i++)
		{
			msg << buffer[i];
		}
		Send(msg);
	}
};

int main(int argc, char* argv[])
{
	CustomClient c;
	c.Connect("127.0.0.1", 60000);

	bool bQuit = false;

	std::cout << "Sending File " << argv[1] << std::endl;
	c.EvaluateBinary((std::string)argv[1]);

	while (!bQuit)
	{

		if (c.IsConnected())
		{
			if (!c.Incoming().empty())
			{


				auto msg = c.Incoming().pop_front().msg;

				switch (msg.header.id)
				{
				case dSHARKMessageType::ServerAccept:
				{
					// Server has responded to a ping request				
					std::cout << "Server Accepted Connection\n";
				}
				break;

				case dSHARKMessageType::EvaluateBinary:
				{
					std::cout << "Binary File Sent" << std::endl;
				}
				break;

				case dSHARKMessageType::ServerPing:
				{
					// Server has responded to a ping request
					std::chrono::system_clock::time_point timeNow = std::chrono::system_clock::now();
					std::chrono::system_clock::time_point timeThen;
					msg >> timeThen;
					std::cout << "Ping: " << std::chrono::duration<double>(timeNow - timeThen).count() << "\n";
				}
				break;

				case dSHARKMessageType::ServerMessage:
				{
					// Server has responded to a ping request	
					uint32_t clientID;
					msg >> clientID;
					std::cout << "Hello from [" << clientID << "]\n";
				}
				break;
				}
			}
		}
		else
		{
			std::cout << "Server Down\n";
			bQuit = true;
		}

	}

	return 0;
}
